import einops
from torch import Tensor, nn

from ..pe import BaseRPE
from ..utils import pe_from_name


class MHA(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        dropout: float = 0.0,
        relative_pe: str | None = None,
        relative_pe_kwargs: dict = {},
    ):
        super(MHA, self).__init__()

        assert embed_size % num_heads == 0, ":("
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        if relative_pe is not None:
            relative_pe_kwargs.setdefault("embed_size", embed_size)
            pe_type = pe_from_name(relative_pe)
            self.relative_pe = (
                pe_type(**relative_pe_kwargs) if issubclass(pe_type, BaseRPE) else None
            )
        else:
            self.relative_pe = None

        self.scale = embed_size**-0.5

    def forward(self, x: Tensor, mask: Tensor | None) -> Tensor:
        Q = einops.rearrange(self.W_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        K = einops.rearrange(self.W_k(x), "b n (h d) -> b h n d", h=self.num_heads)
        V = einops.rearrange(self.W_v(x), "b n (h d) -> b h n d", h=self.num_heads)
        A = einops.einsum(Q, K, "b h i d, b h j d -> b h i j") * self.scale

        if self.relative_pe is not None:
            P = self.relative_pe(x)  # [i j d]
            if self.relative_pe.coupled:
                A += einops.einsum(Q, P, "b h i d, i j d -> b h i j")
            else:
                A += P

        if mask is not None:
            A = A.masked_fill(mask == 0, float("-inf"))

        S = self.dropout(A.softmax(dim=-1))

        O = einops.einsum(S, V, "b h i j, b h j d -> b h i d")

        return self.W_o(einops.rearrange(O, "b h n d -> b n (h d)"))
