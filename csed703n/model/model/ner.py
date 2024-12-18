from typing import Literal

from torch import LongTensor, Tensor, nn

from ..utils import reset_weights
from . import BertBase


class BertNER(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        attn_dropout: float,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        pe_strategy: str,
        pe_kwargs: dict,
        act_fn: str,
    ):
        super(BertNER, self).__init__()

        self.bert = BertBase(
            n_vocab,
            d_model,
            num_heads,
            num_layers,
            d_ff,
            attn_dropout,
            ff_dropout,
            norm,
            pe_strategy,
            pe_kwargs,
            act_fn,
        )

        # TODO: Implement NER head
        self.ner = nn.Identity()

    def reset_weights(
        self, initialization_range: float = 0.02, reset_all: bool = False
    ) -> None:
        if reset_all:
            reset_weights(self.bert, initialization_range)
        reset_weights(self.ner, initialization_range)

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        return self.ner(self.bert(x, mask))
