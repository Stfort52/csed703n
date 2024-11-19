from typing import Literal

from torch import Tensor, nn

from . import Block


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        intermidiate_size: int,
        attn_dropout: float,
        ff_dropout: float,
        norm: Literal["pre", "post"],
        relative_pe: str | None,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermidiate_size = intermidiate_size
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        self.layers = nn.Sequential(
            *[
                Block(
                    embed_size=d_model,
                    num_heads=num_heads,
                    relative_pe=relative_pe,
                    attn_dropout=attn_dropout,
                    intermidiate_size=intermidiate_size,
                    ff_dropout=ff_dropout,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)
