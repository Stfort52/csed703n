import os
import pickle
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from csed703n.data.lightning import GenecorpusDataModule
from csed703n.model.lightning import LightningPretraining
from csed703n.model.model import BertConfig
from csed703n.model.utils import EvenlySpacedModelCheckpoint

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

    dataset_dir = DATA_DIR / "datasets/genecorpus_1M_2048.dataset"
    token_dict = pickle.load((DATA_DIR / "token_dictionary.pkl").open("rb"))

    data = GenecorpusDataModule(
        dataset_dir, token_dict=token_dict, batch_size=BATCH_PER_GPU
    )

    config = BertConfig(
        n_vocab=len(token_dict),
        d_model=256,
        num_heads=4,
        num_layers=6,
        d_ff=512,
        attn_dropout=0.02,
        ff_dropout=0.02,
        norm="post",
        tupe=False,
        absolute_pe_strategy=None,
        absolute_pe_kwargs={},
        relative_pe_strategy="t5",
        relative_pe_kwargs={"num_heads": 4, "num_buckets": 256, "max_distance": 512},
        relative_pe_shared=True,
        act_fn="relu",
    )

    model = LightningPretraining(
        config,
        weight_decay=1e-3,
        lr=1e-3,
        warmup_steps_or_ratio=0.1,
        lr_scheduler="linear",
    )

    checkpoint_callback = EvenlySpacedModelCheckpoint(
        save_last="link", n_ckeckpoints=20
    )
    csv_logger = CSVLogger("checkpoints")
    tb_logger = TensorBoardLogger("checkpoints", version=csv_logger.version)

    trainer = L.Trainer(
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback],
        max_epochs=1,
        strategy="ddp" if WORLD_SIZE > 1 else "auto",
        num_nodes=WORLD_SIZE,
        gradient_clip_val=1.0,
    )

    trainer.print("Start training")
    trainer.print(repr(model))

    trainer.fit(model, data)
