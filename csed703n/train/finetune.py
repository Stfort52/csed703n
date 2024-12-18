import os
import pickle
import random
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from csed703n.data.lightning import NerDataModule
from csed703n.model.lightning import LightningTokenClassification

if __name__ == "__main__":
    torch.random.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_float32_matmul_precision("high")

    WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
    BATCH_SIZE = 32
    BATCH_PER_GPU = BATCH_SIZE // WORLD_SIZE

    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    MODEL_DIR = DATA_DIR.parent / f"checkpoints/lightning_logs/version_{2}"
    TASK_NAME = "bivalent_gene_prediction"

    labels = pd.read_csv(DATA_DIR / "is_bivalent.csv").set_index("id")["H3K27me3"]

    dataset_dir = DATA_DIR / "datasets/panglao_SRA553822-SRS2119548.dataset"
    token_dict = pickle.load((DATA_DIR / "token_dictionary.pkl").open("rb"))

    data = NerDataModule(
        dataset_dir=dataset_dir,
        token_dict=token_dict,
        entity_labels=labels,
        batch_size=BATCH_PER_GPU,
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

    csv_logger = CSVLogger(save_dir, name=TASK_NAME)
    tb_logger = TensorBoardLogger(save_dir, name=TASK_NAME, version=csv_logger.version)
    trainer = L.Trainer(
        strategy="ddp" if WORLD_SIZE > 1 else "auto",
        max_epochs=5,
        logger=[csv_logger, tb_logger],
        num_nodes=WORLD_SIZE,
    )
    trainer.fit(model, data)