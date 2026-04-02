import os
import torch
import torch.nn as nn
import numpy as np
import fire
import wandb
from eb_jepa.logging import get_logger
from eb_jepa.jepa import JEPA
from eb_jepa.architectures import Projector
from eb_jepa.losses import SquareLossSeq, VCLoss  # Cosine: from eb_jepa.losses import CosineLossSeq, VCLoss
from eb_jepa.training_utils import (
    setup_device,
    setup_seed,
    setup_wandb,
    load_config,
    log_epoch,
)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from omegaconf import OmegaConf


from examples.cell_jepa.dataset import CellDataset
from examples.cell_jepa.model import CellEncoder, CellPredictor

logger = get_logger(__name__)

os.environ["WANDB_API_KEY"] = "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"

 
def main(fname="examples/cell_jepa/cfgs/default.yaml", **overrides):
    # Setup
    cfg = load_config(fname, overrides if overrides else None)
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)
    
    # Wandb — skip init if sweep agent already started a run
    _in_sweep = wandb.run is not None
    if not _in_sweep:
        setup_wandb(
            project="eb_jepa_cell",
            config={"example": "cell_jepa", **OmegaConf.to_container(cfg, resolve=True)},
            run_dir=os.path.join(cfg.meta.output_dir, "wandb"),
            enabled=cfg.logging.log_wandb
        )
    _log_wandb = cfg.logging.log_wandb or _in_sweep

    # Data
    full_dataset = CellDataset(cfg.data.h5ad_path, cfg.data.pairs_path)

    # Build valid-transition set from the expert-curated metadata Excel file.
    # Uses ONLY "Developmental progression" edges — same filter as data_pairing.py.
    from collections import defaultdict
    from examples.cell_jepa.data_pairing_utils import parse_developmental_edges
    valid_transitions = defaultdict(set)
    excel_path = cfg.data.get("metadata_excel_path",
                              "/mlbio_scratch/wen2/cellfate_FM/mouse_developmental/metedata.xlsx")
    if full_dataset.cell_type_labels is not None and os.path.exists(excel_path):
        try:
            edges_df = parse_developmental_edges(excel_path)
            name_to_idx = full_dataset.cell_type_to_idx  # cell type name → int index
            for _, row in edges_df.iterrows():
                cx, cy = row["Cell state name (x)"], row["Cell state name (y)"]
                if cx in name_to_idx and cy in name_to_idx:
                    valid_transitions[name_to_idx[cx]].add(name_to_idx[cy])
            n_src = len(valid_transitions)
            n_edges = sum(len(v) for v in valid_transitions.values())
            logger.info(f"Valid transitions (Excel, dev. progression only): {n_src} source types, {n_edges} edges")
        except Exception as e:
            logger.warning(f"Could not load Excel transitions ({e}); falling back to observed pairs.")
            for i_curr, i_next in full_dataset.pairs:
                valid_transitions[full_dataset.cell_type_labels[i_curr]].add(
                    full_dataset.cell_type_labels[i_next])
    else:
        logger.warning("No Excel path found; valid_transitions will be empty.")

    # Reproducible 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # Using generator for reproducibility
    generator = torch.Generator().manual_seed(cfg.meta.seed)
    train_set, test_set = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    # Save test indices for later verification/UMAP
    test_indices = test_set.indices
    os.makedirs(cfg.meta.output_dir, exist_ok=True)
    np.save(os.path.join(cfg.meta.output_dir, "test_indices.npy"), test_indices)

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.data.batch_size, 
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=cfg.data.batch_size, 
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # Models
    input_dim = full_dataset.X.shape[1]
    
    mlp_spec = f"{cfg.model.embed_dim}-{cfg.model.proj_hidden_dim}-{cfg.model.proj_output_dim}"
    projector = Projector(mlp_spec).to(device)
    # Encoder outputs 256D Latent directly
    encoder = CellEncoder(input_dim, cfg.model.embed_dim, cfg.model.hidden_dim).to(device)
    
    # Predictor
    predictor = CellPredictor(cfg.model.embed_dim, pred_hidden_dim=cfg.model.pred_hidden_dim).to(device)
    
    regularizer = VCLoss(
        std_coeff=cfg.loss.std_coeff,
        cov_coeff=cfg.loss.cov_coeff,
        proj=projector
    )

    ploss_fn = SquareLossSeq(proj=None)  # MSE in raw 256D latent space
    # ploss_fn = CosineLossSeq(proj=None)  # Cosine: cosine distance in raw 256D latent space
    jepa = JEPA(
        encoder=encoder,
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=regularizer,
        predcost=ploss_fn
    ).to(device)
    
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    use_cosine = cfg.optim.get("use_cosine", True)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.optim.epochs, eta_min=cfg.optim.lr * 0.01
        )
        logger.info(f"Using cosine LR schedule: {cfg.optim.lr:.2e} → {cfg.optim.lr*0.01:.2e}")
    else:
        scheduler = None
        logger.info(f"Using constant LR: {cfg.optim.lr:.2e}")

    # Training loop
    global_step = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        train_losses = []
        pred_coeff = cfg.loss.get("pred_coeff", 1.0)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            x_curr, x_next = batch[0].to(device), batch[1].to(device)
            x_traj = torch.cat([x_curr, x_next], dim=2)
            
            _, (_, rloss, _, _, ploss) = jepa.unroll(
                observations=x_traj,
                actions=None,
                nsteps=1,
                unroll_mode="autoregressive",
                compute_loss=True,
                stop_grad_target=True,  # detach target — prevents predictor identity collapse
                # stop_grad_target=False,  # OLD: no stop-gradient (target stays in graph)
            )
            
            loss = rloss + pred_coeff * ploss
            
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norms
            grad_l1 = sum(p.grad.detach().abs().sum().item() for p in jepa.parameters() if p.grad is not None)
            grad_l2 = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in jepa.parameters() if p.grad is not None)).item()
            
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "vicreg": f"{rloss.item():.4f}", "pred": f"{ploss.item():.4f}"})
            # pbar.set_postfix({"loss": f"{loss.item():.4f}", "vicreg": f"{rloss.item():.4f}", "cos": f"{ploss.item():.4f}"})  # Cosine
            
            if _log_wandb and global_step % 20 == 0:
                wandb.log({
                    "train/step_total_loss": loss.item(),
                    "train/step_vicreg_loss": rloss.item(),
                    "train/step_prediction_loss": ploss.item(),  # Cosine: "train/step_cosine_loss": ploss.item()
                    "train/grad_norm_l1": grad_l1,
                    "train/grad_norm_l2": grad_l2,
                    "epoch": epoch
                }, step=global_step)
            
            global_step += 1


        # Validation at end of epoch
        jepa.eval()
        test_losses = []
        mse_latent_scores = []
        cos_sim_scores = []
        y_targets = []
        y_currs = []         # y_curr labels for valid-transition metric
        z_targets_list = []
        z_preds_list = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} [Test]"):
                x_curr, x_next = batch[0].to(device), batch[1].to(device)
                y_next = batch[3] if len(batch) > 3 else None  # batch: x_curr, x_next, y_curr, y_next
                x_traj = torch.cat([x_curr, x_next], dim=2)
                
                state, (loss, _, _, _, _) = jepa.unroll(
                    observations=x_traj,
                    actions=None,
                    nsteps=1,
                    unroll_mode="autoregressive",
                    compute_loss=True
                )
                
                z_next_pred = state[:, :, 1].flatten(1)
                z_next_target = jepa.encoder(x_next).flatten(1)
                
                test_losses.append(loss.item())
                mse_latent = torch.mean((z_next_pred - z_next_target)**2, dim=1)
                mse_latent_scores.extend(mse_latent.cpu().numpy().tolist())
                
                # Cosine similarity using vector norms
                cos_sim = (z_next_pred * z_next_target).sum(dim=1) / (torch.norm(z_next_pred, dim=1) * torch.norm(z_next_target, dim=1) + 1e-8)
                cos_sim_scores.extend(cos_sim.cpu().numpy().tolist())
                
                if epoch % 2 == 0 and y_next is not None:
                    y_targets.extend(y_next.numpy().tolist())
                    z_targets_list.append(z_next_target.cpu().numpy())
                    z_preds_list.append(z_next_pred.cpu().numpy())
                    y_currs.extend(batch[2].numpy().tolist())  # y_curr labels needed for valid-transition metric

        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        avg_mse_latent = np.mean(mse_latent_scores) if mse_latent_scores else 0.0
        avg_cos_sim = np.mean(cos_sim_scores) if cos_sim_scores else 0.0
        
        knn_acc = None
        if epoch % 2 == 0 and len(y_targets) > 0 and len(set(y_targets)) > 1:
            from sklearn.neighbors import KNeighborsClassifier
            Z_bank  = np.concatenate(z_targets_list, axis=0)
            Z_query = np.concatenate(z_preds_list,   axis=0)
            Y_next  = np.array(y_targets)
            Y_curr  = np.array(y_currs)
            
            knn = KNeighborsClassifier(n_neighbors=5)  # Euclidean: consistent with MSE training
            # knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")  # Cosine: use with cosine loss
            
            # Predictor KNN: predict cell type of z_next using predicted embedding
            knn.fit(Z_bank, Y_next)
            _, nb_idx = knn.kneighbors(Z_query)
            nb_labels = Y_next[nb_idx]
            knn_acc      = np.mean(knn.predict(Z_query) == Y_next)
            knn_any5_acc = np.mean(np.any(nb_labels == Y_next[:, None], axis=1))

            # Valid-transition metric: count prediction as correct if predicted label
            # is ANY valid next type for the source cell type (not just the specific pair target).
            if valid_transitions:
                pred_labels = knn.predict(Z_query)
                knn_valid_acc = np.mean([
                    pred_labels[i] in valid_transitions[Y_curr[i]]
                    for i in range(len(pred_labels))
                ])
            else:
                knn_valid_acc = None

            logger.info(
                f"Epoch {epoch}: Train={avg_train_loss:.4f} Test={avg_test_loss:.4f} "
                f"MSE={avg_mse_latent:.4f} Cos={avg_cos_sim:.4f} | "
                f"KNN: pred={knn_acc:.3f}/{knn_any5_acc:.3f} "
                f"valid-transition={knn_valid_acc:.3f}"
            )
        else:
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Latent MSE: {avg_mse_latent:.4f}, Latent Cosine Sim: {avg_cos_sim:.4f}")
        
        metrics = {
            "train/epoch_loss": avg_train_loss,
            "test/epoch_loss": avg_test_loss,
            "test/epoch_latent_mse": avg_mse_latent,
            "test/epoch_latent_cos_sim": avg_cos_sim,
            "epoch": epoch
        }
        if knn_acc is not None:
            metrics["test/knn_predictor_acc"]       = knn_acc
            metrics["test/knn_predictor_any5_acc"]  = knn_any5_acc
            if knn_valid_acc is not None:
                metrics["test/knn_predictor_valid_acc"] = knn_valid_acc
        log_epoch(epoch, metrics)
        if _log_wandb:
            wandb.log(metrics, step=global_step)
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.optim.lr
        logger.info(f"  LR after epoch {epoch}: {current_lr:.2e}")
        if _log_wandb:
            wandb.log({"train/lr": current_lr}, step=global_step)

    # Save
    ckpt_path = os.path.join(cfg.meta.output_dir, "cell_jepa.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state_dict": jepa.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Final Visual Validation for WandB
    if _log_wandb:
        logger.info("Generating Final UMAP Overlays for WandB...")
        from examples.cell_jepa.verify_embeddings import verify
        output_dir = os.path.join(cfg.meta.output_dir, "verification")
        
        for metric in ["cosine", "euclidean"]:
            verify(
                ckpt_path=ckpt_path,
                h5ad_path=cfg.data.h5ad_path,
                pairs_path=cfg.data.pairs_path,
                output_dir=output_dir,
                overlay_only=True,
                metric=metric
            )
            img_path = os.path.join(output_dir, f"umap_prediction_overlay_{metric}.png")
            wandb.log({f"val/umap_{metric}": wandb.Image(img_path)})
            logger.info(f"Logged {metric} UMAP to WandB.")

if __name__ == "__main__":
    fire.Fire(main)
