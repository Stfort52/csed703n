from typing import TypedDict

from torch import LongTensor


class Cell(TypedDict):
    input_ids: LongTensor
    length: int
