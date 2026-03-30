import torch
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import os

from examples.cell_jepa.model import CellEncoder, CellPredictor
from examples.cell_jepa.dataset import CellDataset

def verify(ckpt_path, h5ad_path, pairs_path, output_dir, overlay_only=False, metric="cosine", skip_plots=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = OmegaConf.create(ckpt['config'])
    
    # Load Data (Keep X on CPU to save VRAM)
    dataset = CellDataset(h5ad_path, pairs_path)
    X_cpu = torch.from_numpy(dataset.X).float()
    
    from eb_jepa.jepa import JEPA
    from eb_jepa.architectures import Projector

    # Models
    input_dim = X_cpu.shape[1]
    
    # Encoder 
    mlp_spec = f"{cfg.model.embed_dim}-{cfg.model.proj_hidden_dim}-{cfg.model.proj_output_dim}"
    projector = Projector(mlp_spec).to(device)
    encoder = CellEncoder(input_dim, cfg.model.embed_dim, cfg.model.hidden_dim).to(device)
    predictor = CellPredictor(cfg.model.embed_dim, pred_hidden_dim=cfg.model.pred_hidden_dim).to(device)
    
    from eb_jepa.losses import VCLoss, SquareLossSeq
    regularizer = VCLoss(
        std_coeff=cfg.loss.std_coeff, 
        cov_coeff=cfg.loss.cov_coeff, 
        proj=projector
    )
    ploss_fn = SquareLossSeq(proj=projector)
    
    jepa = JEPA(encoder, torch.nn.Identity(), predictor, regularizer, ploss_fn).to(device)
    jepa.load_state_dict(ckpt['model_state_dict'])
    jepa.eval()

    # Compute Embeddings
    with torch.no_grad():
        z_all = []
        bs = 1024
        for i in range(0, len(X_cpu), bs):
            x_batch = X_cpu[i:i+bs].to(device)
            latent = jepa.encoder.backbone(x_batch)
            z_all.append(latent.cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
    
    print(f"Latent shape: {z_all.shape}")

    os.makedirs(output_dir, exist_ok=True)

    if not overlay_only:
        # Raw Background
        print("Computing UMAP on log-normalized gene expression (PCA)...")
        adata_raw = dataset.adata.copy()
        if adata_raw.X.max() > 20: 
            sc.pp.log1p(adata_raw)
        sc.pp.pca(adata_raw, n_comps=50)
        sc.pp.neighbors(adata_raw, use_rep='X_pca')
        sc.tl.umap(adata_raw)
        
        os.makedirs(output_dir, exist_ok=True)
        color_key = 'author_cell_type' if 'author_cell_type' in adata_raw.obs else None
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc.pl.umap(adata_raw, color=color_key, show=False, ax=ax)
        plt.title("Raw Dataset (Log-Normalized Gene Expression)")
        plt.savefig(os.path.join(output_dir, "umap_raw.png"), bbox_inches='tight')
        plt.close()

        # JEPA Latent Space
        adata_latent = dataset.adata.copy()
        adata_latent.obsm['X_jepa'] = z_all
        print("Computing UMAP on JEPA Latent embeddings (256D)...")
        sc.pp.neighbors(adata_latent, use_rep='X_jepa', metric='cosine')
        sc.tl.umap(adata_latent)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sc.pl.umap(adata_latent, color=color_key, show=False, ax=ax)
        plt.title("JEPA Latent Space (256D)")
        plt.savefig(os.path.join(output_dir, "umap_jepa.png"), bbox_inches='tight')
        plt.close()
    
    # Prediction verification (Test Set Split) 
    pairs = dataset.pairs
    test_idx_path = os.path.join(os.path.dirname(output_dir), "test_indices.npy")
    if os.path.exists(test_idx_path):
        print(f"Loading test indices from {test_idx_path}...")
        test_indices = np.load(test_idx_path)
        eval_pairs = pairs[test_indices]
        plot_tag = "Test Set"
    else:
        print("No test indices found, evaluating on full dataset...")
        eval_pairs = pairs
        plot_tag = "Full Dataset"

    idx_curr, idx_next = eval_pairs[:, 0], eval_pairs[:, 1]
    
    
    with torch.no_grad():
        z_curr = torch.from_numpy(z_all[idx_curr]).to(device)
        z_next_target = torch.from_numpy(z_all[idx_next]).to(device)
        
        # Predictor mapping: z_curr -> z_next_pred (256D -> 256D)
        z_next_pred = jepa.predictor(z_curr.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)).flatten(1)
        
        # Compute MSE in Latent Space
        mse_latent_tensor = torch.mean((z_next_pred - z_next_target)**2, dim=1)
        mse_latent = torch.mean(mse_latent_tensor).item()
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(z_next_pred, z_next_target, dim=1)
        avg_cosine = torch.mean(cos_sim).item()

    print(f"{plot_tag} - MSE (Latent): {mse_latent:.4f}, Cosine: {avg_cosine:.4f}")
    if skip_plots:
        print("Skipping all plot generation and saving as requested.")
        return
    
    #TODO: prediction overlay, also include the initial state in the overlay -- would be cool to see the trajectories
    #  Predicted vs True Embedding Overlay --> for test set
    print(f"Generating Predicted vs True UMAP Overlay in 256D space ({plot_tag})...")
    n_overlay = min(500, len(idx_curr))
    n_bg = 10000
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(z_all), size=min(n_bg, len(z_all)), replace=False)
    
    z_overlay_pred = z_next_pred[:n_overlay].cpu().numpy()
    z_overlay_target = z_next_target[:n_overlay].cpu().numpy()
    
    z_joint = np.concatenate([z_all[bg_idx], z_overlay_target, z_overlay_pred], axis=0)
    adata_proj_joint = sc.AnnData(z_joint)
    sc.pp.neighbors(adata_proj_joint, use_rep='X', metric=metric)
    
    sc.tl.umap(adata_proj_joint)
    
    cp_all = adata_proj_joint.obsm['X_umap']
    cp_target = cp_all[len(bg_idx):len(bg_idx)+n_overlay]
    cp_pred = cp_all[len(bg_idx)+n_overlay:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(cp_all[:len(bg_idx), 0], cp_all[:len(bg_idx), 1], c='lightgrey', s=5, alpha=0.3, label='Background')
    plt.scatter(cp_target[:, 0], cp_target[:, 1], c='blue', s=20, label='True Next State')
    plt.scatter(cp_pred[:, 0], cp_pred[:, 1], c='red', s=20, label='Predicted Next State')
    for i in range(n_overlay):
        plt.arrow(cp_target[i, 0], cp_target[i, 1], cp_pred[i, 0]-cp_target[i, 0], cp_pred[i, 1]-cp_target[i, 1], color='black', alpha=0.1)
    
    plt.title(f"UMAP: Predicted vs True ({plot_tag}) | Metric={metric}\nCosine={avg_cosine:.4f}, MSE={mse_latent:.4f}")
    plt.legend()
    fname_overlay = f"umap_prediction_overlay_{metric}.png"
    plt.savefig(os.path.join(output_dir, fname_overlay), bbox_inches='tight')
    plt.close()

    fname_metrics = f"metrics_{metric}.txt"
    with open(os.path.join(output_dir, fname_metrics), "w") as f:
        f.write(f"MSE_Latent: {mse_latent:.4f}\n")
        f.write(f"Cosine_Similarity: {avg_cosine:.4f}\n")

    print(f"Verification artifacts saved to {output_dir}/{fname_metrics} (and UMAP)")

if __name__ == "__main__":
    _kwargs = dict(
        ckpt_path="/mlbio_scratch/kassraie/eb_jepa/output/cell_jepa_balanced_25_25/cell_jepa.pt",
        h5ad_path="/mlbio_scratch/kassraie/World_Model_Cell_Fate/Data_pairs/Pairs/Subset_4/subset4_forward.h5ad",
        pairs_path="/mlbio_scratch/kassraie/World_Model_Cell_Fate/Data_pairs/Pairs/Subset_4/pairs_forward_local.npy",
        output_dir="/mlbio_scratch/kassraie/eb_jepa/output/cell_jepa_balanced_25_25/verification"
    )
    # Run once per metric so two distinct UMAPs + metrics files are produced
    for _metric in ["cosine", "euclidean"]:
        print(f"\n=== Verifying with metric={_metric} ===")
        verify(**_kwargs, metric=_metric)
