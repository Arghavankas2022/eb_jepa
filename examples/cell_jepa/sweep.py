"""
W&B Sweeps for Cell JEPA.

Two sweep configs available:
  - vc_sweep:  grid over std_coeff × cov_coeff
  - lr_sweep:  grid over lr × use_cosine  (std=10, cov=50 fixed)

Usage:
  # Register a sweep and get a SWEEP_ID:
  python -m examples.cell_jepa.sweep register --sweep_name=lr_sweep

  # Launch agent(s) in tmux or terminal:
  python -m examples.cell_jepa.sweep agent <SWEEP_ID>
"""

import os
import wandb
import fire

os.environ["WANDB_API_KEY"] = "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"

WANDB_PROJECT = "eb_jepa_cell"
BASE_CFG      = "examples/cell_jepa/cfgs/subset4.yaml"

# ── Sweep configs ────────────────────────────────────────────────────────────

SWEEP_CONFIGS = {
    "vc_sweep": {
        "name":   "cell_jepa_vc_sweep",
        "method": "grid",
        "metric": {"name": "test/knn_predictor_any5_acc", "goal": "maximize"},
        "parameters": {
            "std_coeff":  {"values": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]},
            "cov_coeff":  {"values": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            "lr":         {"value":  0.0005},
            "use_cosine": {"value":  True},
        },
    },
    "lr_sweep": {
        "name":   "cell_jepa_lr_sweep",
        "method": "grid",
        "metric": {"name": "test/knn_predictor_any5_acc", "goal": "maximize"},
        "parameters": {
            "std_coeff":  {"value":  10.0},
            "cov_coeff":  {"value":  50.0},
            "lr":         {"values": [0.00001, 0.00005, 0.0001, 0.0003, 0.0005, 0.001]},
            "use_cosine": {"values": [True, False]},
        },
    },
}

# ─────────────────────────────────────────────────────────────────────────────


def _train():
    """Called by the W&B agent for each run in the sweep."""
    run = wandb.init()
    cfg = run.config

    std        = cfg.std_coeff
    cov        = cfg.cov_coeff
    lr         = cfg.lr
    use_cosine = cfg.use_cosine

    tag        = f"std{std}_cov{cov}_lr{lr}_cos{use_cosine}".replace(".", "_")
    output_dir = f"output/sweep/{tag}_{run.id}"

    # main() detects wandb.run is active and skips re-init
    from examples.cell_jepa.main import main
    main(
        fname=BASE_CFG,
        **{
            "meta.output_dir":    output_dir,
            "loss.std_coeff":     std,
            "loss.cov_coeff":     cov,
            "optim.lr":           lr,
            "optim.use_cosine":   use_cosine,
            "logging.log_wandb":  False,  # sweep agent owns the run
        }
    )
    wandb.finish()


def register(sweep_name: str = "lr_sweep"):
    """Register a sweep on W&B and print the sweep ID.
    
    Args:
        sweep_name: one of 'vc_sweep' or 'lr_sweep'
    """
    if sweep_name not in SWEEP_CONFIGS:
        print(f"Unknown sweep '{sweep_name}'. Choose from: {list(SWEEP_CONFIGS.keys())}")
        return
    sweep_id = wandb.sweep(SWEEP_CONFIGS[sweep_name], project=WANDB_PROJECT)
    print(f"\n✅ Sweep '{sweep_name}' registered: {sweep_id}")
    print(f"\nTo launch an agent (in tmux for persistence):")
    print(f"  tmux new-session -d -s sweep_agent \"cd $(pwd) && conda run -n eb_jepa python -m examples.cell_jepa.sweep agent {sweep_id}\"")
    print(f"\nOr directly:")
    print(f"  python -m examples.cell_jepa.sweep agent {sweep_id}\n")


def agent(sweep_id: str, count: int = None):
    """Launch a W&B agent that picks up runs from the sweep."""
    print(f"Starting agent for sweep: {sweep_id}")
    wandb.agent(sweep_id, function=_train, project=WANDB_PROJECT, count=count)


if __name__ == "__main__":
    fire.Fire({"register": register, "agent": agent})
