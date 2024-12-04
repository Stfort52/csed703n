from typing import Type

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


def pe_from_name(name: str) -> Type[BasePE]:
    match name.lower().split("-"):
        case pe_type, pe_kind:
            return MODEL_MAP[pe_type][pe_kind]
        case _:
            raise ValueError(f"Unknown PE type: {name}")
