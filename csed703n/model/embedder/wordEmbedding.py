from torch import LongTensor, Tensor, nn


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, ln_eps=1e-12):
        super(WordEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layer_norm = nn.LayerNorm(embed_size, ln_eps)

    def forward(self, x: LongTensor) -> Tensor:
        return self.layer_norm(self.embed(x))
