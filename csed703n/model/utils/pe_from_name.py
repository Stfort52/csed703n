from typing import Literal, Type, overload

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


@overload
def pe_from_name(category: Literal["absolute"], name: str) -> Type[BasePE]: ...


@overload
def pe_from_name(category: Literal["relative"], name: str) -> Type[BaseRPE]: ...


def pe_from_name(category: str, name: str) -> Type[BasePE]:
    return MODEL_MAP[category][name]
