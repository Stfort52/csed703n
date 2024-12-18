import einops
from torch import Tensor, nn


class TUPE(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super(TUPE, self).__init__()

        assert embed_size % num_heads == 0, ":("
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.U_q = nn.Linear(embed_size, embed_size)
        self.U_k = nn.Linear(embed_size, embed_size)

        self.scale = (2 * self.head_dim) ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        Q = einops.rearrange(self.U_q(x), "n (h d) -> h n d", h=self.num_heads)
        K = einops.rearrange(self.U_k(x), "n (h d) -> h n d", h=self.num_heads)
        A = einops.einsum(Q, K, "h i d, h j d -> h i j") * self.scale

        return A
