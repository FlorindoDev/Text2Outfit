"""Small PyTorch loop with resumable LoRA/MLP checkpoints every epoch."""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.outfit.text2outfit import LOSS_NAMES, Text2OutfitModel
from .config import TrainingConfig

LOGGER = logging.getLogger("training.seed_to_outfit")
LOG_EVERY = 50


def _save(payload: Any, path: Path, *, json_file: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        if json_file:
            temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def training_contract(model: Text2OutfitModel) -> dict[str, Any]:
    llm = model.generator.llm
    return dict(
        categories=list(model.generator.categories),
        subcategories=[list(labels) for labels in model.subcategories],
        seed_mode=model.seed_mode, loss_weights=list(model.loss_weights), temperature=model.temperature,
        llm_name=str(getattr(llm, "model_path", type(llm).__name__)),
        lora=getattr(llm, "lora_settings", None), compatibility=model.compatibility.source,
        config=json.loads(json.dumps(asdict(model.config), default=str)),
        architecture=model.architecture(),
    )


def restore_checkpoint(
    path: str | Path, model: Text2OutfitModel, optimizer: AdamW,
    train_loader: DataLoader, *, expected_run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_schema_version") != 1:
        raise ValueError("unsupported seed-to-outfit checkpoint schema")
    if checkpoint["model_config"] != training_contract(model):
        raise ValueError("checkpoint backbone, CP model, taxonomy or training objective does not match")
    if expected_run_config is not None and checkpoint["run_config"] != expected_run_config:
        raise ValueError("resume requires the same dataset and model configuration")
    state = checkpoint["model_state_dict"]
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("checkpoint must contain every trainable LoRA/MLP/query parameter")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or expected.intersection(incompatible.missing_keys):
        raise ValueError("invalid training state dictionary")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    random.setstate(checkpoint["rng_state"]["python"])
    torch.set_rng_state(checkpoint["rng_state"]["torch"])
    if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
    if train_loader.generator is not None and checkpoint.get("loader_rng_state") is not None:
        train_loader.generator.set_state(checkpoint["loader_rng_state"])
    return checkpoint


def train(
    model: Text2OutfitModel, train_loader: DataLoader,
    validation_loader: DataLoader, config: TrainingConfig,
    *, run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .plots import save_loss_plots

    config.validate()
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("train and validation loaders cannot be empty")
    output = Path(config.output_dir)
    if config.resume is None and ((output / "best.pt").exists() or list((output / "epochs").glob("*.pt"))):
        raise FileExistsError(f"run already exists: {output}; choose another output directory or resume")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    optimizer = AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    metadata = json.loads(json.dumps(run_config or {}, default=str))
    history: dict[str, Any] = dict(epochs=[], train=[], validation=[])
    start_epoch, best_loss, global_step = 1, float("inf"), 0
    if config.resume is not None:
        checkpoint = restore_checkpoint(config.resume, model, optimizer, train_loader, expected_run_config=metadata)
        history = checkpoint["training_history"]
        start_epoch, best_loss = checkpoint["epoch"] + 1, checkpoint["best_validation_loss"]
        global_step = checkpoint["global_step"]
        previous = checkpoint["training_config"]
        for name in ("learning_rate", "weight_decay", "gradient_accumulation_steps", "max_grad_norm"):
            if previous[name] != getattr(config, name):
                raise ValueError(f"resume cannot change {name}")
        if any(int(path.stem.rsplit("_", 1)[-1]) >= start_epoch for path in (output / "epochs").glob("seed_to_outfit_epoch_*.pt")):
            raise FileExistsError("later epochs already exist; choose a new output directory for this resume")
        if start_epoch > config.epochs:
            raise ValueError("resume epoch already reaches the requested number of epochs")
    _save(dict(training=asdict(config), model=training_contract(model), run=metadata), output / "config.json", json_file=True)
    (output / "architecture.txt").write_text(model.architecture_summary(), encoding="utf-8")

    for epoch in range(start_epoch, config.epochs + 1):
        LOGGER.info("epoch=%d/%d phase=train batches=%d", epoch, config.epochs, len(train_loader))
        model.train()
        sums = dict.fromkeys(LOSS_NAMES, 0.0)
        examples = 0
        group_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            losses = model.compute_losses(batch)
            size = len(batch.prompts)
            if not all(bool(torch.isfinite(loss).all()) for loss in losses.values()):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}, batch {index}")
            # Accumulate sums, then divide by the actual number of examples (also for the last group).
            (losses["total"] * size).backward()
            examples += size
            for name in LOSS_NAMES:
                sums[name] += float(losses[name].detach()) * size
            group_examples += size
            if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(train_loader):
                for parameter in parameters:
                    if parameter.grad is not None:
                        parameter.grad.div_(group_examples)
                nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                group_examples = 0
                global_step += 1
            if (index + 1) % LOG_EVERY == 0 or index + 1 == len(train_loader):
                LOGGER.info(
                    "epoch=%d microbatch=%d/%d loss=%.6f lr=%.8g phase=train",
                    epoch, index + 1, len(train_loader),
                    float(losses["total"].detach()), optimizer.param_groups[0]["lr"],
                )
        train_metrics = {name: value / examples for name, value in sums.items()}
        LOGGER.info("epoch=%d/%d phase=validation batches=%d", epoch, config.epochs, len(validation_loader))
        model.eval()
        sums = dict.fromkeys(LOSS_NAMES, 0.0)
        examples = 0
        with torch.no_grad():
            for index, batch in enumerate(validation_loader, start=1):
                losses = model.compute_losses(batch)
                if not all(bool(torch.isfinite(loss).all()) for loss in losses.values()):
                    raise FloatingPointError(f"non-finite validation loss at epoch {epoch}")
                size = len(batch.prompts)
                examples += size
                for name in LOSS_NAMES:
                    sums[name] += float(losses[name]) * size
                if index % LOG_EVERY == 0 or index == len(validation_loader):
                    LOGGER.info(
                        "epoch=%d microbatch=%d/%d loss=%.6f lr=%.8g phase=validation",
                        epoch, index, len(validation_loader),
                        float(losses["total"]), optimizer.param_groups[0]["lr"],
                    )
        validation_metrics = {name: value / examples for name, value in sums.items()}
        history["epochs"].append(epoch)
        history["train"].append(train_metrics)
        history["validation"].append(validation_metrics)
        improved = validation_metrics["total"] < best_loss
        best_loss = min(best_loss, validation_metrics["total"])
        payload = dict(
            checkpoint_schema_version=1, epoch=epoch, global_step=global_step,
            state_dict_kind="trainable_parameters",
            model_state_dict={name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad},
            optimizer_state_dict=optimizer.state_dict(),
            model_config=training_contract(model),
            training_config=json.loads(json.dumps(asdict(config), default=str)),
            run_config=metadata, training_history=history, best_validation_loss=best_loss,
            train_metrics=train_metrics, validation_metrics=validation_metrics,
            rng_state=dict(python=random.getstate(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
            loader_rng_state=train_loader.generator.get_state() if train_loader.generator is not None else None,
        )
        epoch_path = output / "epochs" / f"seed_to_outfit_epoch_{epoch:03d}.pt"
        _save(payload, epoch_path)
        if improved:
            temporary = output / "best.pt.tmp"
            shutil.copyfile(epoch_path, temporary)
            temporary.replace(output / "best.pt")
        _save(history, output / "history.json", json_file=True)
        save_loss_plots(history, output / "plots", epoch)
        LOGGER.info(
            "epoch=%d/%d train_loss=%.6f val_loss=%.6f best_val_loss=%.6f",
            epoch, config.epochs, train_metrics["total"], validation_metrics["total"], best_loss,
        )
    return history
