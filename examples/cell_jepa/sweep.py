"""
W&B Sweep for Cell JEPA — std_coeff × cov_coeff grid.

Usage:
  1. Register the sweep and get a SWEEP_ID:
       python -m examples.cell_jepa.sweep register

  2. Launch one or more agents (can run on multiple machines simultaneously):
       python -m examples.cell_jepa.sweep agent <SWEEP_ID>
"""

import os
import sys
import wandb
import fire

os.environ["WANDB_API_KEY"] = "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"

WANDB_PROJECT = "eb_jepa_cell"
BASE_CFG      = "examples/cell_jepa/cfgs/subset4.yaml"

SWEEP_CONFIG = {
    "name":   "cell_jepa_vc_sweep",
    "method": "grid",
    "metric": {"name": "test/knn_predictor_acc", "goal": "maximize"},
    "parameters": {
        "std_coeff": {"values": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]},
        "cov_coeff": {"values": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
    },
}


def _train():
    """Called by the W&B agent for each run in the sweep."""
    run = wandb.init()
    std = run.config.std_coeff
    cov = run.config.cov_coeff

    tag        = f"std{std}_cov{cov}".replace(".", "_")
    output_dir = f"output/sweep/{tag}_{run.id}"

    # main() will detect wandb.run is already active and skip re-init
    from examples.cell_jepa.main import main
    main(
        fname=BASE_CFG,
        **{
            "meta.output_dir":   output_dir,
            "loss.std_coeff":    std,
            "loss.cov_coeff":    cov,
            "logging.log_wandb": False,  # main won't re-init; _in_sweep handles logging
        }
    )
    wandb.finish()


def register():
    """Register the sweep on W&B and print the sweep ID."""
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_PROJECT)
    print(f"\n✅ Sweep registered: {sweep_id}")
    print(f"\nTo launch an agent, run:")
    print(f"  python -m examples.cell_jepa.sweep agent {sweep_id}\n")


def agent(sweep_id: str, count: int = None):
    """Launch a W&B agent that picks up runs from the sweep."""
    print(f"Starting agent for sweep: {sweep_id}")
    wandb.agent(sweep_id, function=_train, project=WANDB_PROJECT, count=count)


if __name__ == "__main__":
    fire.Fire({"register": register, "agent": agent})
