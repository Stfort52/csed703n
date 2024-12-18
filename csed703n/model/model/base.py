from typing import Literal

from torch import LongTensor, Tensor, nn

from ..embedder import WordEmbedding
from ..pe import BaseRPE
from ..transformer import Encoder
from ..utils import pe_from_name


class BertBase(nn.Module):
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
        super(BertBase, self).__init__()
        self.embedder = WordEmbedding(n_vocab, d_model, dropout_p=ff_dropout)

        pe = pe_from_name(pe_strategy)
        pe_kwargs.setdefault("embed_size", d_model)
        self.pe = pe(**pe_kwargs) if not issubclass(pe, BaseRPE) else None

        self.encoder = Encoder(
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

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        if self.pe is not None:
            x = self.embedder(x) + self.pe(x)
        else:
            x = self.embedder(x)
        return self.encoder(x, mask)
