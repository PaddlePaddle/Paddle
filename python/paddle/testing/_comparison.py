#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import abc
import cmath
import collections.abc
import contextlib
from typing import TYPE_CHECKING, Any, Callable, NoReturn

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

import paddle
import paddle.framework

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]


class ErrorMeta(Exception):
    """Internal testing exception that makes that carries error metadata."""

    def __init__(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ()
    ) -> None:
        super().__init__(
            "If you are a user and see this message during normal operation, "
            "it implies a mismatch found by paddle.testing.assert_close."
        )
        self.type = type
        self.msg = msg
        self.id = id

    def to_error(
        self, msg: str | Callable[[str], str] | None = None
    ) -> Exception:
        if not isinstance(msg, str):
            generated_msg = self.msg
            if self.id:
                generated_msg += f"\n\nThe failure occurred for item {''.join(str([item]) for item in self.id)}"

            msg = msg(generated_msg) if callable(msg) else generated_msg

        return self.type(msg)


# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    paddle.float16: (0.001, 1e-5),
    paddle.float32: (1.3e-6, 1e-5),
    paddle.float64: (1e-7, 1e-7),
    paddle.complex64: (1.3e-6, 1e-5),
    paddle.complex128: (1e-7, 1e-7),
}

# Add bfloat16 if available in current paddle version
if hasattr(paddle, "bfloat16"):
    _DTYPE_PRECISIONS[paddle.bfloat16] = (0.016, 1e-5)

# Handle quantized types if they map to standard dtypes in comparison logic
# For simplicity, we default integers to exact match (rtol=0, atol=0) usually,
# but if handled as floats (e.g. dequantized), we use float32 rules.
_QUANTIZED_TYPES = [
    getattr(paddle, t) for t in ["int8", "int16", "uint8"] if hasattr(paddle, t)
]
for q_type in _QUANTIZED_TYPES:
    # Using float32 tolerances as a fallback if they are compared as floats
    _DTYPE_PRECISIONS[q_type] = _DTYPE_PRECISIONS[paddle.float32]


def default_tolerances(
    *inputs: paddle.Tensor | paddle.dtype,
    dtype_precisions: dict[paddle.dtype, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Returns the default absolute and relative testing tolerances."""
    dtypes = []
    for input in inputs:
        if isinstance(input, paddle.Tensor):
            dtypes.append(input.dtype)
        elif isinstance(
            input, (paddle.dtype, paddle.framework.core.VarDesc.VarType)
        ):
            dtypes.append(input)
        else:
            raise TypeError(
                f"Expected a paddle.Tensor or a paddle.dtype, but got {type(input)} instead."
            )
    dtype_precisions = dtype_precisions or _DTYPE_PRECISIONS
    rtols, atols = zip(
        *[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes]
    )
    return max(rtols), max(atols)


def get_tolerances(
    *inputs: paddle.Tensor | paddle.dtype,
    rtol: float | None,
    atol: float | None,
    id: tuple[Any, ...] = (),
) -> tuple[float, float]:
    """Gets absolute and relative to be used for numeric comparisons."""
    if (rtol is None) ^ (atol is None):
        raise ErrorMeta(
            ValueError,
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
            id=id,
        )
    elif rtol is not None and atol is not None:
        return rtol, atol
    else:
        return default_tolerances(*inputs)


def _make_bitwise_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    first_mismatch_idx: tuple[int] | None = None,
):
    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not 'equal'!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"
    if first_mismatch_idx is not None:
        msg += (
            f"The first mismatched element is at index {first_mismatch_idx}.\n"
        )
    return msg.strip()


def _make_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    abs_diff: float,
    abs_diff_idx: int | tuple[int, ...] | None = None,
    atol: float,
    rel_diff: float,
    rel_diff_idx: int | tuple[int, ...] | None = None,
    rtol: float,
) -> str:
    equality = rtol == 0 and atol == 0

    def make_diff_msg(
        *,
        type: str,
        diff: float,
        idx: int | tuple[int, ...] | None,
        tol: float,
    ) -> str:
        if idx is None:
            msg = f"{type.title()} difference: {diff}"
        else:
            msg = f"Greatest {type} difference: {diff} at index {idx}"
        if not equality:
            msg += f" (up to {tol} allowed)"
        return msg + "\n"

    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"

    msg += make_diff_msg(
        type="absolute", diff=abs_diff, idx=abs_diff_idx, tol=atol
    )
    msg += make_diff_msg(
        type="relative", diff=rel_diff, idx=rel_diff_idx, tol=rtol
    )

    return msg.strip()


def make_scalar_mismatch_msg(
    actual: bool | complex,
    expected: bool | complex,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
) -> str:
    abs_diff = abs(actual - expected)
    rel_diff = float("inf") if expected == 0 else abs_diff / abs(expected)
    return _make_mismatch_msg(
        default_identifier="Scalars",
        identifier=identifier,
        extra=f"Expected {expected} but got {actual}.",
        abs_diff=abs_diff,
        atol=atol,
        rel_diff=rel_diff,
        rtol=rtol,
    )


def make_tensor_mismatch_msg(
    actual: paddle.Tensor,
    expected: paddle.Tensor,
    matches: paddle.Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
):
    def unravel_flat_index(flat_index: int) -> tuple[int, ...]:
        if not matches.shape:
            return ()

        inverse_index = []
        for size in matches.shape[::-1]:
            div, mod = divmod(flat_index, size)
            flat_index = div
            inverse_index.append(mod)

        return tuple(inverse_index[::-1])

    number_of_elements = matches.numel().item()
    total_mismatches = number_of_elements - int(
        paddle.sum(matches.astype("int64")).item()
    )
    extra = (
        f"Mismatched elements: {total_mismatches} / {number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%})"
    )

    # Floating point low precision checks (skipped for brevity/paddle specifics, using standard logic)

    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    matches_flat = matches.flatten()

    if (
        actual.dtype
        not in [
            paddle.float16,
            paddle.float32,
            paddle.float64,
            paddle.complex64,
            paddle.complex128,
        ]
        and hasattr(paddle, "bfloat16")
        and actual.dtype != paddle.bfloat16
    ):
        # Cast to int64 for non-floating types to avoid overflow in diff calc
        actual_flat = actual_flat.astype("int64")
        expected_flat = expected_flat.astype("int64")

    abs_diff = paddle.abs(actual_flat - expected_flat)

    # Ensure that only mismatches are used for the max_abs_diff computation
    # Note: paddle doesn't support bool indexing assignment cleanly in all versions, using where
    abs_diff = paddle.where(matches_flat, paddle.zeros_like(abs_diff), abs_diff)

    # max returns (value,), argsort or manual search for index
    max_abs_diff = paddle.max(abs_diff)
    max_abs_diff_flat_idx = paddle.argmax(abs_diff)

    rel_diff = abs_diff / paddle.abs(expected_flat)
    # Ensure that only mismatches are used for the max_rel_diff computation
    rel_diff = paddle.where(matches_flat, paddle.zeros_like(rel_diff), rel_diff)

    max_rel_diff = paddle.max(rel_diff)
    max_rel_diff_flat_idx = paddle.argmax(rel_diff)

    return _make_mismatch_msg(
        default_identifier="Tensor-likes",
        identifier=identifier,
        extra=extra,
        abs_diff=max_abs_diff.item(),
        abs_diff_idx=unravel_flat_index(int(max_abs_diff_flat_idx)),
        atol=atol,
        rel_diff=max_rel_diff.item(),
        rel_diff_idx=unravel_flat_index(int(max_rel_diff_flat_idx)),
        rtol=rtol,
    )


class UnsupportedInputs(Exception):
    """Exception to be raised during the construction of a :class:`Pair` in case it doesn't support the inputs."""


class Pair(abc.ABC):
    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        **unknown_parameters: Any,
    ) -> None:
        self.actual = actual
        self.expected = expected
        self.id = id
        self._unknown_parameters = unknown_parameters

    @staticmethod
    def _inputs_not_supported() -> NoReturn:
        raise UnsupportedInputs

    @staticmethod
    def _check_inputs_isinstance(*inputs: Any, cls: type | tuple[type, ...]):
        if not all(isinstance(input, cls) for input in inputs):
            Pair._inputs_not_supported()

    def _fail(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ()
    ) -> NoReturn:
        raise ErrorMeta(
            type, msg, id=self.id if not id and hasattr(self, "id") else id
        )

    @abc.abstractmethod
    def compare(self) -> None:
        """Compares the inputs and raises an :class`ErrorMeta` in case they mismatch."""

    def extra_repr(self) -> Sequence[str | tuple[str, Any]]:
        return []

    def __repr__(self) -> str:
        head = f"{type(self).__name__}("
        tail = ")"
        body = [
            f"    {name}={value!s},"
            for name, value in [
                ("id", self.id),
                ("actual", self.actual),
                ("expected", self.expected),
                *[
                    (extra, getattr(self, extra))
                    if isinstance(extra, str)
                    else extra
                    for extra in self.extra_repr()
                ],
            ]
        ]
        return "\n".join((head, *body, *tail))


class ObjectPair(Pair):
    """Pair for any type of inputs that will be compared with the `==` operator."""

    def compare(self) -> None:
        try:
            equal = self.actual == self.expected
        except Exception as error:
            raise ErrorMeta(
                ValueError,
                f"{self.actual} == {self.expected} failed with:\n{error}.",
                id=self.id,
            ) from error

        if not equal:
            self._fail(AssertionError, f"{self.actual} != {self.expected}")


class NonePair(Pair):
    """Pair for ``None`` inputs."""

    def __init__(
        self, actual: Any, expected: Any, **other_parameters: Any
    ) -> None:
        if not (actual is None or expected is None):
            self._inputs_not_supported()

        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> None:
        if not (self.actual is None and self.expected is None):
            self._fail(
                AssertionError,
                f"None mismatch: {self.actual} is not {self.expected}",
            )


class BooleanPair(Pair):
    """Pair for :class:`bool` inputs."""

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...],
        **other_parameters: Any,
    ) -> None:
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, **other_parameters)

    @property
    def _supported_types(self) -> tuple[type, ...]:
        cls: list[type] = [bool]
        if HAS_NUMPY:
            cls.append(np.bool_)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[bool, bool]:
        self._check_inputs_isinstance(
            actual, expected, cls=self._supported_types
        )
        actual, expected = (
            self._to_bool(bool_like, id=id) for bool_like in (actual, expected)
        )
        return actual, expected

    def _to_bool(self, bool_like: Any, *, id: tuple[Any, ...]) -> bool:
        if isinstance(bool_like, bool):
            return bool_like
        elif isinstance(bool_like, np.bool_):
            return bool_like.item()
        else:
            raise ErrorMeta(
                TypeError, f"Unknown boolean type {type(bool_like)}.", id=id
            )

    def compare(self) -> None:
        if self.actual is not self.expected:
            self._fail(
                AssertionError,
                f"Booleans mismatch: {self.actual} is not {self.expected}",
            )


class NumberPair(Pair):
    """Pair for Python number inputs."""

    _TYPE_TO_DTYPE = {
        int: paddle.int64,
        float: paddle.float64,
        complex: paddle.complex128,
    }
    _NUMBER_TYPES = tuple(_TYPE_TO_DTYPE.keys())

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_dtype: bool = False,
        **other_parameters: Any,
    ) -> None:
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            *[self._TYPE_TO_DTYPE[type(input)] for input in (actual, expected)],
            rtol=rtol,
            atol=atol,
            id=id,
        )
        self.equal_nan = equal_nan
        self.check_dtype = check_dtype

    @property
    def _supported_types(self) -> tuple[type, ...]:
        cls = list(self._NUMBER_TYPES)
        if HAS_NUMPY:
            cls.append(np.number)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[int | float | complex, int | float | complex]:
        self._check_inputs_isinstance(
            actual, expected, cls=self._supported_types
        )
        actual, expected = (
            self._to_number(number_like, id=id)
            for number_like in (actual, expected)
        )
        return actual, expected

    def _to_number(
        self, number_like: Any, *, id: tuple[Any, ...]
    ) -> int | float | complex:
        if HAS_NUMPY and isinstance(number_like, np.number):
            return number_like.item()
        elif isinstance(number_like, self._NUMBER_TYPES):
            return number_like  # type: ignore[return-value]
        else:
            raise ErrorMeta(
                TypeError, f"Unknown number type {type(number_like)}.", id=id
            )

    def compare(self) -> None:
        if self.check_dtype and type(self.actual) is not type(self.expected):
            self._fail(
                AssertionError,
                f"The (d)types do not match: {type(self.actual)} != {type(self.expected)}.",
            )

        if self.actual == self.expected:
            return

        if (
            self.equal_nan
            and cmath.isnan(self.actual)
            and cmath.isnan(self.expected)
        ):
            return

        abs_diff = abs(self.actual - self.expected)
        tolerance = self.atol + self.rtol * abs(self.expected)

        if cmath.isfinite(abs_diff) and abs_diff <= tolerance:
            return

        self._fail(
            AssertionError,
            make_scalar_mismatch_msg(
                self.actual, self.expected, rtol=self.rtol, atol=self.atol
            ),
        )

    def extra_repr(self) -> Sequence[str]:
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_dtype",
        )


class TensorLikePair(Pair):
    """Pair for :class:`paddle.Tensor`-like inputs.

    Note: Paddle's sparse and quantized tensor APIs differ from PyTorch's.
    This implementation focuses on standard Strided tensors.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        allow_subclasses: bool = True,
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_device: bool = True,
        check_dtype: bool = True,
        check_layout: bool = False,
        check_stride: bool = False,
        **other_parameters: Any,
    ):
        actual, expected = self._process_inputs(
            actual, expected, id=id, allow_subclasses=allow_subclasses
        )
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            actual, expected, rtol=rtol, atol=atol, id=self.id
        )
        self.equal_nan = equal_nan
        self.check_device = check_device
        self.check_dtype = check_dtype
        self.check_layout = check_layout
        self.check_stride = check_stride

    def _process_inputs(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...],
        allow_subclasses: bool,
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        directly_related = isinstance(actual, type(expected)) or isinstance(
            expected, type(actual)
        )
        if not directly_related:
            self._inputs_not_supported()

        if not allow_subclasses and type(actual) is not type(expected):
            self._inputs_not_supported()

        actual, expected = (
            self._to_tensor(input) for input in (actual, expected)
        )
        return actual, expected

    def _to_tensor(self, tensor_like: Any) -> paddle.Tensor:
        if isinstance(tensor_like, paddle.Tensor):
            return tensor_like

        try:
            return paddle.to_tensor(tensor_like)
        except Exception:
            self._inputs_not_supported()

    def compare(self) -> None:
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        # Skip value check for meta tensors? Paddle doesn't strictly have a 'meta' device type exposed like Torch
        # but if shapes mismatch we already raised.

        actual, expected = self._equalize_attributes(actual, expected)
        self._compare_values(actual, expected)

    def _compare_attributes(
        self,
        actual: paddle.Tensor,
        expected: paddle.Tensor,
    ) -> None:
        def raise_mismatch_error(
            attribute_name: str, actual_value: Any, expected_value: Any
        ) -> NoReturn:
            self._fail(
                AssertionError,
                f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.",
            )

        if tuple(actual.shape) != tuple(expected.shape):
            raise_mismatch_error("shape", actual.shape, expected.shape)

        if (
            self.check_layout
            and hasattr(actual, 'layout')
            and hasattr(expected, 'layout')
        ):
            if actual.layout != expected.layout:
                raise_mismatch_error("layout", actual.layout, expected.layout)

        if self.check_device:
            # Paddle places (CPUPlace, CUDAPlace) compare strictly.
            # Note: str(actual.place) helps debugging
            if str(actual.place) != str(expected.place):
                raise_mismatch_error("place", actual.place, expected.place)

        if self.check_dtype and actual.dtype != expected.dtype:
            raise_mismatch_error("dtype", actual.dtype, expected.dtype)

    def _equalize_attributes(
        self, actual: paddle.Tensor, expected: paddle.Tensor
    ) -> tuple[paddle.Tensor, paddle.Tensor]:
        # Normalize places to CPU if they mismatch or if we need to do numpy/scalar ops
        # Simplification: if places mismatch, move both to CPU
        if str(actual.place) != str(expected.place):
            actual = actual.cpu()
            expected = expected.cpu()

        if actual.dtype != expected.dtype:
            # Basic manual type promotion
            # If one is float64 and other is float32, go to float64
            # If complex involved, go to complex
            # This is a simplification compared to torch.promote_types
            actual_is_fp = actual.dtype in [
                paddle.float32,
                paddle.float64,
                paddle.float16,
            ]
            expected_is_fp = expected.dtype in [
                paddle.float32,
                paddle.float64,
                paddle.float16,
            ]

            target_dtype = actual.dtype
            if (
                expected.dtype == paddle.complex128
                or actual.dtype == paddle.complex128
            ):
                target_dtype = paddle.complex128
            elif (
                expected.dtype == paddle.complex64
                or actual.dtype == paddle.complex64
            ):
                target_dtype = (
                    paddle.complex64
                    if target_dtype != paddle.complex128
                    else paddle.complex128
                )
            elif (
                expected.dtype == paddle.float64
                or actual.dtype == paddle.float64
            ):
                target_dtype = paddle.float64
            elif (
                expected.dtype == paddle.float32
                or actual.dtype == paddle.float32
            ):
                target_dtype = paddle.float32

            # Simple cast
            actual = actual.astype(target_dtype)
            expected = expected.astype(target_dtype)

        return actual, expected

    def _compare_values(
        self, actual: paddle.Tensor, expected: paddle.Tensor
    ) -> None:
        # For simplicity in this port, we stick to standard regular value comparison.
        # Specialized checks for Sparse/Quantized are routed here after conversion in PyTorch,
        # but Paddle's ecosystem handles these differently.
        self._compare_regular_values_close(
            actual,
            expected,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=self.equal_nan,
        )

    def _compare_regular_values_close(
        self,
        actual: paddle.Tensor,
        expected: paddle.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
        identifier: str | Callable[[str], str] | None = None,
    ) -> None:
        """Checks if the values of two tensors are close up to a desired tolerance."""
        # paddle.isclose returns a boolean tensor
        matches = paddle.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

        if paddle.all(matches):
            return

        if actual.shape == []:
            msg = make_scalar_mismatch_msg(
                actual.item(),
                expected.item(),
                rtol=rtol,
                atol=atol,
                identifier=identifier,
            )
        else:
            msg = make_tensor_mismatch_msg(
                actual,
                expected,
                matches,
                rtol=rtol,
                atol=atol,
                identifier=identifier,
            )
        self._fail(AssertionError, msg)

    def extra_repr(self) -> Sequence[str]:
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_device",
            "check_dtype",
            "check_layout",
            "check_stride",
        )


def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]],
    sequence_types: tuple[type, ...] = (collections.abc.Sequence,),
    mapping_types: tuple[type, ...] = (collections.abc.Mapping,),
    id: tuple[Any, ...] = (),
    **options: Any,
) -> list[Pair]:
    # We explicitly exclude str's here since they are self-referential
    if (
        isinstance(actual, sequence_types)
        and not isinstance(actual, str)
        and isinstance(expected, sequence_types)
        and not isinstance(expected, str)
    ):
        actual_len = len(actual)  # type: ignore[arg-type]
        expected_len = len(expected)  # type: ignore[arg-type]
        if actual_len != expected_len:
            raise ErrorMeta(
                AssertionError,
                f"The length of the sequences mismatch: {actual_len} != {expected_len}",
                id=id,
            )

        pairs = []
        for idx in range(actual_len):
            pairs.extend(
                originate_pairs(
                    actual[idx],  # type: ignore[index]
                    expected[idx],  # type: ignore[index]
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, idx),
                    **options,
                )
            )
        return pairs

    elif isinstance(actual, mapping_types) and isinstance(
        expected, mapping_types
    ):
        actual_keys = set(actual.keys())  # type: ignore[attr-defined]
        expected_keys = set(expected.keys())  # type: ignore[attr-defined]
        if actual_keys != expected_keys:
            missing_keys = expected_keys - actual_keys
            additional_keys = actual_keys - expected_keys
            raise ErrorMeta(
                AssertionError,
                (
                    f"The keys of the mappings do not match:\n"
                    f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
                    f"Additional keys in the actual mapping: {sorted(additional_keys)}"
                ),
                id=id,
            )

        keys: Collection = actual_keys
        with contextlib.suppress(Exception):
            keys = sorted(keys)

        pairs = []
        for key in keys:
            pairs.extend(
                originate_pairs(
                    actual[key],  # type: ignore[index]
                    expected[key],  # type: ignore[index]
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, key),
                    **options,
                )
            )
        return pairs

    else:
        for pair_type in pair_types:
            try:
                return [pair_type(actual, expected, id=id, **options)]
            except UnsupportedInputs:
                continue
            except ErrorMeta:
                raise
            except Exception as error:
                raise RuntimeError(
                    f"Originating a {pair_type.__name__}() at item {''.join(str([item]) for item in id)} with\n\n"
                    f"{type(actual).__name__}(): {actual}\n\n"
                    f"and\n\n"
                    f"{type(expected).__name__}(): {expected}\n\n"
                    f"resulted in the unexpected exception above. "
                ) from error
        else:
            raise ErrorMeta(
                TypeError,
                f"No comparison pair was able to handle inputs of type {type(actual)} and {type(expected)}.",
                id=id,
            )


def not_close_error_metas(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]] = (ObjectPair,),
    sequence_types: tuple[type, ...] = (collections.abc.Sequence,),
    mapping_types: tuple[type, ...] = (collections.abc.Mapping,),
    **options: Any,
) -> list[ErrorMeta]:
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    try:
        pairs = originate_pairs(
            actual,
            expected,
            pair_types=pair_types,
            sequence_types=sequence_types,
            mapping_types=mapping_types,
            **options,
        )
    except ErrorMeta as error_meta:
        raise error_meta.to_error() from None

    error_metas: list[ErrorMeta] = []
    for pair in pairs:
        try:
            pair.compare()
        except ErrorMeta as error_meta:
            error_metas.append(error_meta)
        except Exception as error:
            raise RuntimeError(
                f"Comparing\n\n"
                f"{pair}\n\n"
                f"resulted in the unexpected exception above."
            ) from error

    # Break cycles
    error_metas = [error_metas]
    return error_metas.pop()


def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    msg: str | Callable[[str], str] | None = None,
):
    r"""Asserts that ``actual`` and ``expected`` are close.

    This is a PaddlePaddle implementation inspired by ``torch.testing.assert_close``.

    If ``actual`` and ``expected`` are tensors, they are considered close if:

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default), inputs of directly related types are allowed.
        rtol (Optional[float]): Relative tolerance.
        atol (Optional[float]): Absolute tolerance.
        equal_nan (Union[bool, str]): If ``True``, two ``NaN`` values will be considered equal.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same Place.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``.
        check_layout (bool): If ``True`` (default), asserts layout matches (if attribute exists).
        msg (Optional[Union[str, Callable[[str], str]]]): Optional error message.
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    error_metas = not_close_error_metas(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        msg=msg,
    )

    if error_metas:
        raise error_metas[0].to_error(msg)
