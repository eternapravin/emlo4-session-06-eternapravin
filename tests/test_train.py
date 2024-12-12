import os
from pathlib import Path
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from typing import List

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)


@task_wrapper
def model_test(
    cfg: DictConfig,
    trainer: L.Trainer,
    model: L.LightningModule,
    datamodule: L.LightningDataModule,
):
    log.info("Starting testing!")
    if cfg.ckpt_path:
        log.info(f"Loading checkpoint: {cfg.ckpt_path}")
        test_metrics = trainer.test(model, datamodule, ckpt_path=cfg.ckpt_path)
    else:
        log.warning("No checkpoint path provided! Using current model weights.")
        test_metrics = trainer.test(model, datamodule)
    
    log.info(f"Test metrics:\n{test_metrics}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Set up paths
    print(OmegaConf.to_yaml(cfg))
    log_dir = Path(cfg.paths.log_dir)

    # Set up logger
    setup_logger(log_dir / "test_log.log")

    # Initialize DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    print("Printing of DataModule")
    print(OmegaConf.to_yaml(cfg.data))

    # Initialize Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    print("Printing of ModelConfig")
    print(OmegaConf.to_yaml(cfg.model))

    # Set up callbacks (optional, only if required for testing)
    callbacks: List[L.Callback] = []
    if cfg.get("callbacks"):
        from src.train import instantiate_callbacks  # Import from train script
        callbacks = instantiate_callbacks(cfg.callbacks)
    print("Printing of CallbackConfigs")
    print(OmegaConf.to_yaml(cfg.get("callbacks", {})))

    # Initialize Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
    )

    # Perform Testing
    model_test(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()
