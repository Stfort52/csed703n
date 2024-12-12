from typing import Literal, TypedDict


class BertConfig(TypedDict):
    n_vocab: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    attn_dropout: float
    ff_dropout: float
    norm: Literal["pre", "post"]
    pe_strategy: str
    pe_kwargs: dict
