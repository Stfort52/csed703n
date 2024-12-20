import os
import pickle
import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from csed703n.data.lightning import NerDataModule
from csed703n.model.lightning import LightningTokenClassification


def main(version: int, seed: int = 42):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_float32_matmul_precision("high")

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    BATCH_SIZE = 32
    BATCH_PER_GPU = BATCH_SIZE // WORLD_SIZE

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    MODEL_DIR = DATA_DIR.parent / f"checkpoints/lightning_logs/version_{version}"
    TASK_NAME = "tf_range_prediction"

    labels = pd.read_csv(DATA_DIR / "is_longrange_tf.csv").set_index("id")[
        "is_longrange"
    ]

    dataset_dir = DATA_DIR / "datasets/iCM_diff_dropseq.dataset"
    token_dict = pickle.load((DATA_DIR / "token_dictionary.pkl").open("rb"))

    data = NerDataModule(
        dataset_dir=dataset_dir,
        token_dict=token_dict,
        entity_labels=labels,
        batch_size=BATCH_PER_GPU,
        dataset_shuffle=42,
        label_shuffle=42,
    )

    ckpt_dir = MODEL_DIR / "checkpoints" / "last.ckpt"
    save_dir = MODEL_DIR / "finetune"

    model = LightningTokenClassification(
        model_path_or_config=ckpt_dir,
        n_classes=labels.nunique(),
        lr=5e-5,
        weight_decay=1e-3,
        lr_scheduler="linear",
        warmup_steps_or_ratio=0.1,
        freeze_first_n_layers=0,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", every_n_epochs=1
    )
    csv_logger = CSVLogger(save_dir, name=TASK_NAME)
    tb_logger = TensorBoardLogger(save_dir, name=TASK_NAME, version=csv_logger.version)
    trainer = L.Trainer(
        strategy="ddp" if WORLD_SIZE > 1 else "auto",
        max_epochs=3,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        num_nodes=WORLD_SIZE,
    )
    trainer.fit(model, data)

    # load best checkpoint
    best_model = LightningTokenClassification.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )


if __name__ == "__main__":
    from itertools import product

    seeds = [42, 424, 4242, 42424, 424242]
    versions = [6]

    for version, seed in product(versions, seeds):
        print(f"Version: {version}, Seed: {seed}")
        main(version, seed)
