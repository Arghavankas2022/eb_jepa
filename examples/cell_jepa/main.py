import os
import torch
import torch.nn as nn
import numpy as np
import fire
import wandb
from eb_jepa.logging import get_logger
from eb_jepa.jepa import JEPA
from eb_jepa.architectures import Projector
from eb_jepa.losses import SquareLossSeq, VCLoss
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

#TODO: add cell type classification as a downstream task, and evaluate the model on accuracy of this 
# use k-neigherest neighbour and use some voting to pick the nearest cell types, (?even if one of them is correct, still a win)

 
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

    ploss_fn = SquareLossSeq(proj=projector)
    jepa = JEPA(
        encoder=encoder,
        aencoder=nn.Identity(),
        predictor=predictor,
        regularizer=regularizer,
        predcost=ploss_fn
    ).to(device)
    
    optimizer = torch.optim.AdamW(jepa.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    
    # Training loop
    global_step = 0
    for epoch in range(cfg.optim.epochs):
        jepa.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            x_curr, x_next = batch[0].to(device), batch[1].to(device)
            x_traj = torch.cat([x_curr, x_next], dim=2)
            
            _, (loss, rloss, _, _, ploss) = jepa.unroll(
                observations=x_traj,
                actions=None,
                nsteps=1,
                unroll_mode="autoregressive",
                compute_loss=True
            )
            
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norms
            grad_l1 = sum(p.grad.detach().abs().sum().item() for p in jepa.parameters() if p.grad is not None)
            grad_l2 = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in jepa.parameters() if p.grad is not None)).item()
            
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "pred": f"{ploss.item():.4f}"})
            
            if _log_wandb and global_step % 20 == 0:
                wandb.log({
                    "train/step_total_loss": loss.item(),
                    "train/step_vicreg_loss": rloss.item(),
                    "train/step_prediction_loss": ploss.item(),
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
        y_currs = []         # for identity baseline
        z_targets_list = []
        z_preds_list = []
        z_currs_list = []    # for static encoder baseline
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} [Test]"):
                x_curr, x_next = batch[0].to(device), batch[1].to(device)
                y_next = batch[2] if len(batch) > 2 else None
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
                    # Baselines
                    z_curr_enc = jepa.encoder(x_curr).flatten(1)
                    z_currs_list.append(z_curr_enc.cpu().numpy())
                    y_currs.extend(batch[2].numpy().tolist())  # y_curr = label of x_curr

        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        avg_mse_latent = np.mean(mse_latent_scores) if mse_latent_scores else 0.0
        avg_cos_sim = np.mean(cos_sim_scores) if cos_sim_scores else 0.0
        
        knn_acc = None
        knn_static_acc = None
        knn_identity_acc = None
        if epoch % 2 == 0 and len(y_targets) > 0 and len(set(y_targets)) > 1:
            from sklearn.neighbors import KNeighborsClassifier
            Z_bank   = np.concatenate(z_targets_list, axis=0)
            Z_query  = np.concatenate(z_preds_list,   axis=0)
            Z_curr   = np.concatenate(z_currs_list,   axis=0)
            Y_next   = np.array(y_targets)
            Y_curr   = np.array(y_currs)
            
            knn = KNeighborsClassifier(n_neighbors=5)
            
            # Predictor KNN: predict cell type of z_next using predicted embedding
            knn.fit(Z_bank, Y_next)
            knn_acc = np.mean(knn.predict(Z_query) == Y_next)
            
            # Static encoder baseline: use z_curr (no predictor) to predict y_next
            knn.fit(Z_bank, Y_next)
            knn_static_acc = np.mean(knn.predict(Z_curr) == Y_next)
            
            # Identity baseline: use z_curr to predict y_curr (same-state discrimination)
            knn.fit(Z_curr, Y_curr)
            knn_identity_acc = np.mean(knn.predict(Z_curr) == Y_curr)
            
            logger.info(
                f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
                f"Latent MSE: {avg_mse_latent:.4f}, Cosine: {avg_cos_sim:.4f} | "
                f"KNN predictor={knn_acc:.4f} static_enc={knn_static_acc:.4f} identity={knn_identity_acc:.4f}"
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
            metrics["test/knn_predictor_acc"]  = knn_acc
            metrics["test/knn_static_enc_acc"] = knn_static_acc
            metrics["test/knn_identity_acc"]   = knn_identity_acc
        log_epoch(epoch, metrics)
        if _log_wandb:
            wandb.log(metrics, step=global_step)

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
