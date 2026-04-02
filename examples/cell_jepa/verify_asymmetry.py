import torch
import numpy as np
import os
from omegaconf import OmegaConf
from examples.cell_jepa.model import CellEncoder, CellPredictor
from examples.cell_jepa.dataset import CellDataset
import matplotlib.pyplot as plt
import pandas as pd

def get_metrics(jepa, dataset, device, batch_size=1024):
    X_cpu = torch.from_numpy(dataset.X).float()
    pairs = dataset.pairs
    
    with torch.no_grad():
        # Encode all cells to 256D Latent
        z_all = []
        for i in range(0, len(X_cpu), batch_size):
            x_batch = X_cpu[i:i+batch_size].to(device)
            z = jepa.encoder.backbone(x_batch)
            z_all.append(z.cpu().numpy())
        z_all = np.concatenate(z_all, axis=0)
        
        # Extract pairs
        idx_curr, idx_next = pairs[:, 0], pairs[:, 1]
        z_curr = torch.from_numpy(z_all[idx_curr]).to(device)
        z_next_target = torch.from_numpy(z_all[idx_next]).to(device)
        
        # Predict transitions
        # Input to predictor: [B, D, T, 1, 1] -> here [B, 256, 1, 1, 1]
        z_next_pred = jepa.predictor(z_curr.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)).flatten(1)
        
        # Compute Metrics
        mse = torch.mean((z_next_pred - z_next_target)**2, dim=1).cpu().numpy()
        cos_sim = torch.nn.functional.cosine_similarity(z_next_pred, z_next_target, dim=1).cpu().numpy()
            
    return {
        "MSE": np.mean(mse),
        "Cosine": np.mean(cos_sim),
    }

def run_asymmetry_test(ckpt_path, forward_h5ad, forward_pairs, backward_h5ad, backward_pairs, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = OmegaConf.create(ckpt['config'])
    
    from eb_jepa.jepa import JEPA
    from eb_jepa.architectures import Projector
    from eb_jepa.losses import VCLoss, SquareLossSeq

    f_dataset = CellDataset(forward_h5ad, forward_pairs)
    input_dim = f_dataset.X.shape[1]
    
    mlp_spec = f"{cfg.model.embed_dim}-{cfg.model.proj_hidden_dim}-{cfg.model.proj_output_dim}"
    projector = Projector(mlp_spec).to(device)
    encoder = CellEncoder(input_dim, cfg.model.embed_dim, cfg.model.hidden_dim).to(device)
    predictor = CellPredictor(cfg.model.embed_dim, pred_hidden_dim=cfg.model.pred_hidden_dim).to(device)
    
    regularizer = VCLoss(
        std_coeff=cfg.loss.std_coeff, 
        cov_coeff=cfg.loss.cov_coeff, 
        proj=projector
    )
    ploss_fn = SquareLossSeq(proj=None)  # prediction loss in raw 256D latent space
    
    jepa = JEPA(encoder, torch.nn.Identity(), predictor, regularizer, ploss_fn).to(device)
    jepa.load_state_dict(ckpt['model_state_dict'])
    jepa.eval()
    
    # Load test indices for forward dataset
    test_idx_path = os.path.join(os.path.dirname(ckpt_path), "test_indices.npy")
    if os.path.exists(test_idx_path):
        f_test_indices = np.load(test_idx_path)
        f_dataset.pairs = f_dataset.pairs[f_test_indices]
        print(f"Filtered forward dataset to test set ({len(f_test_indices)} pairs).")

    print("\n--- Evaluating Forward Transitions (The 'Arrow of Time') ---")
    f_metrics = get_metrics(jepa, f_dataset, device)
    print(f"Forward MSE:      {f_metrics['MSE']:.4f}")
    print(f"Forward Cosine:   {f_metrics['Cosine']:.4f}")
    
    print("\n--- Evaluating Backward Transitions (The 'Impossible Path') ---")
    b_dataset = CellDataset(backward_h5ad, backward_pairs)
    
    # Create test split for backward dataset using identical seed logic to ensure fair comparison
    from torch.utils.data import random_split
    b_train_size = int(0.8 * len(b_dataset.pairs))
    b_test_size = len(b_dataset.pairs) - b_train_size
    generator = torch.Generator().manual_seed(cfg.meta.seed)
    _, b_test_set = random_split(range(len(b_dataset.pairs)), [b_train_size, b_test_size], generator=generator)
    b_dataset.pairs = b_dataset.pairs[b_test_set.indices]
    print(f"Filtered backward dataset to test set ({len(b_test_set.indices)} pairs).")

    b_metrics = get_metrics(jepa, b_dataset, device)
    print(f"Backward MSE:     {b_metrics['MSE']:.4f}")
    print(f"Backward Cosine:  {b_metrics['Cosine']:.4f}")
    
    # Summary Table
    df = pd.DataFrame([f_metrics, b_metrics], index=["Forward", "Backward"])
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "asymmetry_report.csv"))
    
    # Plot Comparison
    metrics_to_plot = ["MSE", "Cosine"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, m in enumerate(metrics_to_plot):
        axes[i].bar(["Forward", "Backward"], [f_metrics[m], b_metrics[m]], color=['blue', 'red'])
        axes[i].set_title(m)
        if m == "MSE": axes[i].set_ylabel("Value (Lower is Better)")
        else: axes[i].set_ylabel("Value (Higher is Better)")
        
    plt.suptitle("JEPA World Model: Directional Asymmetry Test (Fate vs Reverse Fate)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "asymmetry_comparison.png"))
    plt.close()
    
    print(f"\nAsymmetry test complete. Report saved to {output_dir}")
    
    # Asymmetry Ratios (Forward vs Backward)
    mse_ratio = b_metrics['MSE'] / (f_metrics['MSE'] + 1e-8)
    cos_ratio = f_metrics['Cosine'] / (b_metrics['Cosine'] + 1e-8)

    print("\n--- Directional Asymmetry Summary ---")
    print(f"MSE Asymmetry Index (Backward / Forward): {mse_ratio:.2f}x (Higher is better)")
    print(f"Cosine Asymmetry Index (Forward / Backward): {cos_ratio:.2f}x (Higher is better)")
    
    if mse_ratio > 1.2 or cos_ratio > 1.2:
        print("\nSUCCESS: The model shows a clear biological 'Arrow of Time'.")
        print("It understands that Forward transitions are mathematically much more plausible than Backward transitions.")
    else:
        print("\nWARNING: The model lacks clear asymmetry. It may just be modeling cell proximity.")

if __name__ == "__main__":
    import fire
    fire.Fire(run_asymmetry_test)
