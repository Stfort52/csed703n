from torch import Tensor, nn


class LanguageModeling(nn.Module):
    def __init__(self, d_model: int, n_vocab: int):
        super(LanguageModeling, self).__init__()

        self.dense = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.unembedding = nn.Linear(d_model, n_vocab, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [b, n, d]
        return self.unembedding(self.dense(x))

    def tie_weights(self, embeddings: nn.Embedding):
        self.unembedding.weight = embeddings.weight
