from pathlib import Path
from typing import cast

import datasets
import lightning as L
from torch.utils.data import DataLoader

from ..utils import MlmDataCollator


class GenecorpusDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str | Path,
        token_dict: dict[str, int],
        batch_size=32,
        shuffle: int | bool = 42,
        num_workers=4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = Path(dataset_dir)
        self.token_dict = token_dict
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        self.dataset = cast(datasets.Dataset, datasets.load_from_disk(self.dataset_dir))
        self.dataset.set_format(
            type="torch", columns=["input_ids"], output_all_columns=True
        )

        match self.shuffle:
            case int():
                self.dataset = self.dataset.shuffle(seed=self.shuffle)
            case True:
                self.dataset = self.dataset.shuffle(seed=None)
            case False:
                pass

        self.collator = MlmDataCollator(
            token_dict=self.token_dict,
            pad_token_or_index="<pad>",
            mask_token_or_index="<mask>",
            mask_prob=0.15,
        )

    def train_dataloader(self):
        assert self.dataset is not None

        return DataLoader(
            self.dataset,  # pyright: ignore[reportArgumentType]
            batch_size=self.batch_size,
            collate_fn=self.collator,  # pyright: ignore[reportArgumentType]
            shuffle=False,
            num_workers=self.num_workers,
        )