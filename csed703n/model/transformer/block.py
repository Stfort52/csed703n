from typing import Literal

from torch import Tensor, nn

from . import MHA


class Block(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        attn_dropout: float,
        intermidiate_size: int,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        relative_pe: str | None = None,
        relative_pe_kwargs: dict = {},
    ):
        super(Block, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.intermidiate_size = intermidiate_size
        self.norm = norm

        self.attn = MHA(
            embed_size, num_heads, attn_dropout, relative_pe, relative_pe_kwargs
        )

        self.ln1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, intermidiate_size),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(intermidiate_size, embed_size),
        )
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm == "pre":
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        elif self.norm == "post":
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ff(x))
        else:
            raise ValueError(":(")

        return x
