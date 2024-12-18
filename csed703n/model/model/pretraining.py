from typing import Literal

from torch import LongTensor, Tensor, nn

from ..embedder import WordEmbedding
from ..pe import BaseRPE
from ..transformer import Encoder
from ..unembedder import LanguageModeling
from ..utils import pe_from_name


class BertPretraining(nn.Module):
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
    ):
        super(BertPretraining, self).__init__()
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
        )

        self.lm = LanguageModeling(d_model, n_vocab)
        self.lm.tie_weights(self.embedder.embed)

    def reset_weights(self, initializer_range: float = 0.02):
        for module in self.modules():
            match module:
                case nn.Linear():
                    module.weight.data.normal_(mean=0.0, std=initializer_range)
                    if module.bias is not None:
                        module.bias.data.zero_()
                case nn.Embedding():
                    module.weight.data.normal_(mean=0.0, std=initializer_range)
                case nn.LayerNorm():
                    module.weight.data.fill_(1.0)
                    module.bias.data.zero_()
                case _:
                    pass

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        if self.pe is not None:
            x = self.embedder(x) + self.pe(x)
        else:
            x = self.embedder(x)
        x = self.encoder(x, mask)
        return self.lm(x)
