from torch import LongTensor, Tensor

from ..pe import BasePE
from ..transformer import TUPE
from . import BertBase


class TupeBase(BertBase):
    """
    The TUPE model from the paper
    "Rethinking Positional Encoding in Language Pre-training"
    (https://arxiv.org/abs/2006.15595)
    """

    def __init__(self, *args, **kwargs):
        super(TupeBase, self).__init__(*args, **kwargs)
        self.tupe = TUPE(kwargs["d_model"], kwargs["num_heads"])
        self.absolute_pe: BasePE

    def _check_args(self) -> None:
        assert (
            self.absolute_pe_strategy is not None
        ), "TUPE requires absolute positional encodings"

    def forward(self, x: LongTensor, mask: LongTensor | None = None) -> Tensor:
        p = self.tupe(self.absolute_pe(x))
        x = self.embedder(x)
        return self.encoder(x, mask, p)
