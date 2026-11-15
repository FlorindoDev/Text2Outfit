"""Cumulative train/validation loss plots in the OutfitTransformer style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from model.outfit.text2outfit import LOSS_NAMES


def save_loss_plots(history: dict, output_dir: str | Path, epoch: int) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    overview, panels = plt.subplots(3, 2, figsize=(20, 18))
    try:
        for name, panel in zip(LOSS_NAMES, panels.flat):
            figure, axes = plt.subplots(figsize=(10, 6))
            try:
                for axis in (axes, panel):
                    axis.plot(history["epochs"], [row[name] for row in history["train"]], marker="o", label="Train loss")
                    axis.plot(history["epochs"], [row[name] for row in history["validation"]], marker="o", label="Validation loss")
                    axis.set(title=f"Seed-to-outfit {name} loss", xlabel="Epoch", ylabel="Loss")
                    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
                    axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                    axis.grid(alpha=0.3)
                    axis.legend()
                figure.tight_layout()
                figure.savefig(directory / f"{name}_loss_epoch_{epoch:03d}.png")
            finally:
                plt.close(figure)
        overview.tight_layout()
        overview.savefig(directory / f"losses_epoch_{epoch:03d}.png")
    finally:
        plt.close(overview)
