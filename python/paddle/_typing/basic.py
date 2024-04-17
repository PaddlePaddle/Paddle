from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, Union, Sequence

import numpy as np

from .. import Tensor

Numberic: TypeAlias = Union[int, float, complex, np.number[Any], Tensor]

_T = TypeVar("_T", bound=Numberic)
_SeqLevel1: TypeAlias = Sequence[_T]
_SeqLevel2: TypeAlias = Sequence[Sequence[_T]]
_SeqLevel3: TypeAlias = Sequence[Sequence[Sequence[_T]]]
_SeqLevel4: TypeAlias = Sequence[Sequence[Sequence[Sequence[_T]]]]
_SeqLevel5: TypeAlias = Sequence[Sequence[Sequence[Sequence[Sequence[_T]]]]]
_SeqLevel6: TypeAlias = Sequence[
    Sequence[Sequence[Sequence[Sequence[Sequence[_T]]]]]
]

IntSequence: TypeAlias = _SeqLevel1[int]

NumbericSequence: TypeAlias = _SeqLevel1[Numberic]

NestedSequence: TypeAlias = Union[
    _T,
    _SeqLevel1[_T],
    _SeqLevel2[_T],
    _SeqLevel3[_T],
    _SeqLevel4[_T],
    _SeqLevel5[_T],
    _SeqLevel6[_T],
]

NestedNumbericSequence: TypeAlias = NestedSequence[Numberic]

TensorOrTensors: TypeAlias = Union[Tensor, Sequence[Tensor]]
