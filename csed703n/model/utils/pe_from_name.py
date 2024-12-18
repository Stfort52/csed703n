from typing import Literal, Type

from ..pe import *

MODEL_MAP = {
    "absolute": {
        "trained": TrainedPE,
        "sinusoidal": SinusoidalPE,
    },
    "relative": {
        "trained": TrainedRPE,
        "sinusoidal": SinusoidalRPE,
    },
}


def pe_from_name(category: Literal["absolute", "relative"], name: str) -> Type[BasePE]:
    return MODEL_MAP[category][name]
