from torch import LongTensor, Tensor, nn


class WordEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        padding_idx: int | None = None,
        dropout_p: float = 0.0,
        ln_eps: float = 1e-12,
    ):
        super(WordEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.layer_norm = nn.LayerNorm(embed_size, ln_eps)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: LongTensor) -> Tensor:
        return self.layer_norm(self.dropout(self.embed(x)))
