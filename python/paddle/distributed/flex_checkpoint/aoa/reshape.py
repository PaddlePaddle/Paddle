# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
"""The AoA ``reshape`` primitive.

A reshape keeps the row-major order of the elements and only changes where the
rows break. Every plan entry, though, is "one rectangular region (a *box*) of
the source -> one box of the destination", because read requests, receive
buffers and slice copies all describe boxes. So a reshape has to cut each
region into *pieces* that are boxes on both sides; such a piece holds the same
elements in the same order on both sides and moves as "slice it out, reshape
it". This module finds those pieces.

One reshape: the flat index is cut at every inner block size of either shape
that the other shape can cut at too; between two neighbouring cuts lies a
*factor*, a coordinate both shapes share. Each shape writes a factor's index
as a mixed-radix number of its own -- one digit on both sides when the shapes
merely group axes differently (``(6, 4)`` and ``(2, 3, 4)`` share the factors
2, 3 and 4), several where an axis boundary of one shape falls inside an axis
of the other (``(6, 4)`` and ``(4, 6)`` share one factor of 24, written 6 x 4
and 4 x 6). A region is a box of a shape exactly when, on every axis and on
every factor, the digits it takes cover one contiguous range -- checked one
axis or factor at a time, which keeps the common layouts at one piece.

A chain: a slice ref's data goes through its postprocess list -- permutes,
casts and reshapes, each run on the data itself -- so a piece of the ref must
be a box at every stage. Every reshape is cut with its own mapping, and a
piece of one reshape that another reshape cuts is cut again, until every stage
agrees.

The module has four sections, each using only the ones above it:

1. limits and errors;
2. geometry of one reshape (:class:`Box`, :class:`MixedRadixAxis`,
   :class:`FactoredShape`, :class:`ReshapeMapping`) -- integers only;
3. :class:`PostprocessChain` -- the stages of one postprocess list;
4. slice refs -- what ``AOAEngine`` calls.
"""

from __future__ import annotations

import ast
import functools
import itertools
import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from .postprocess import (
    invert_permutation,
    is_permute,
    is_reshape,
    make_reshape_op,
    reshape_op_shape,
    transpose_list,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .aoa_engine import SliceRef

# ---------------------------------------------------------------------------
# 1. Limits and errors
# ---------------------------------------------------------------------------

# Cutting an axis that the other side lays out in many separate stretches
# costs one piece per stretch, so one region can fall into millions of pieces.
# Past this many it is refused instead of enumerated.
MAX_RESHAPE_PIECES = 1024


class ReshapeError(ValueError):
    """A reshape statement the engine refuses."""


class IncompatibleShapesError(ReshapeError):
    """The two shapes hold different numbers of elements."""


class TooManyPiecesError(ReshapeError):
    """A region would fall into more than ``MAX_RESHAPE_PIECES`` pieces."""


class ReshapeAttributeError(ReshapeError):
    """The ``reshape=`` attribute does not fit the statement."""


def _message(requirement: str, reason: str, **context: object) -> str:
    described = ", ".join(f"{name}={value}" for name, value in context.items())
    return f"{requirement}: {reason}. Got {described}."


class _PieceLimitReached(Exception):
    """Internal: raised by the geometry, turned into TooManyPiecesError."""


# ---------------------------------------------------------------------------
# 2. Geometry of one reshape
# ---------------------------------------------------------------------------


class Box(NamedTuple):
    """One range per axis: ``offsets[i] : offsets[i] + sizes[i]``."""

    offsets: tuple[int, ...]
    sizes: tuple[int, ...]

    @classmethod
    def from_slices(cls, slices: Sequence[slice]) -> Box:
        # Slice refs and shard slices are always built with integer bounds and
        # unit step (``make_input_tensor``, ``concat``, ``split``,
        # ``find_shard_sources`` and :meth:`to_slices`), so an open bound or a
        # stride here is a bug in whatever built the slice.
        for sl in slices:
            assert (
                isinstance(sl.start, numbers.Integral)
                and isinstance(sl.stop, numbers.Integral)
                and sl.step in (None, 1)
            ), f"expected integer bounds and unit step, got {sl} in {slices}"
        return cls(
            tuple(sl.start for sl in slices),
            tuple(sl.stop - sl.start for sl in slices),
        )

    def to_slices(self) -> tuple[slice, ...]:
        return tuple(
            slice(offset, offset + size, 1)
            for offset, size in zip(self.offsets, self.sizes)
        )

    @classmethod
    def concat(cls, parts: Sequence[Box]) -> Box:
        """The box whose axes are those of ``parts``, one after another."""
        return cls(
            tuple(itertools.chain.from_iterable(p.offsets for p in parts)),
            tuple(itertools.chain.from_iterable(p.sizes for p in parts)),
        )

    def select(self, axes: range) -> Box:
        """The box restricted to ``axes``."""
        return Box(
            self.offsets[axes.start : axes.stop],
            self.sizes[axes.start : axes.stop],
        )

    def permuted(self, permutation: Sequence[int]) -> Box:
        """The box with its axes in the order of ``permutation``."""
        return Box(
            tuple(transpose_list(self.offsets, permutation)),
            tuple(transpose_list(self.sizes, permutation)),
        )

    def relative_to(self, origin: Box) -> Box:
        """This box in coordinates whose zero is the corner of ``origin``."""
        return Box(
            tuple(o - b for o, b in zip(self.offsets, origin.offsets)),
            self.sizes,
        )

    def offset_by(self, origin: Box) -> Box:
        """The inverse of :meth:`relative_to`."""
        return Box(
            tuple(o + b for o, b in zip(self.offsets, origin.offsets)),
            self.sizes,
        )

    def intersect(self, other: Box) -> Box | None:
        """The common region, or ``None`` if it is empty."""
        offsets, sizes = [], []
        for o1, s1, o2, s2 in zip(
            self.offsets, self.sizes, other.offsets, other.sizes
        ):
            start, stop = max(o1, o2), min(o1 + s1, o2 + s2)
            if start >= stop:
                return None
            offsets.append(start)
            sizes.append(stop - start)
        return Box(tuple(offsets), tuple(sizes))


def _product(
    per_part: Sequence[Sequence[Box]], max_pieces: int | None
) -> list[Box]:
    """Every combination of one box per part, each as one box over all the
    parts; refused before anything is built when there would be too many."""
    if max_pieces is not None and (
        math.prod(len(boxes) for boxes in per_part) > max_pieces
    ):
        raise _PieceLimitReached
    return [Box.concat(parts) for parts in itertools.product(*per_part)]


@dataclass(frozen=True)
class MixedRadixAxis:
    """An index read as a mixed-radix number: an axis of a shape, or a factor
    as one shape writes it.

    ``digits`` are most significant first: axis 0 of ``(6, 4)`` against
    ``(2, 3, 4)`` has digits ``(2, 3)`` and index ``i`` has digits
    ``(i // 3, i % 3)``. A size-one axis has no digits. Boxes passed to and
    returned by these methods are boxes over the digits; a *range* is a
    contiguous range ``[start, start + length)`` of the index itself.
    """

    digits: tuple[int, ...]

    @functools.cached_property
    def strides(self) -> list[int]:
        strides = [1] * len(self.digits)
        for i in range(len(self.digits) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.digits[i + 1]
        return strides

    def boxes_of_range(self, start: int, length: int) -> list[Box]:
        """The digit boxes that together cover one range of the axis.

        Takes the widest box that starts at the current index and fits, then
        continues, so a range that already is a digit box yields just that
        box. ``(2, 3)``, range ``[2, 6)`` -> ``[0:1, 2:3]`` and ``[1:2, 0:3]``.
        """
        if not self.digits:
            return [Box((), ())]
        digits, strides = self.digits, self.strides
        boxes = []
        while length > 0:
            level = next(
                i
                for i, stride in enumerate(strides)
                if start % stride == 0 and stride <= length
            )
            stride = strides[level]
            count = min(
                length // stride,
                digits[level] - (start // stride) % digits[level],
            )
            boxes.append(
                Box(
                    tuple(
                        (start // strides[i]) % digits[i] if i <= level else 0
                        for i in range(len(digits))
                    ),
                    (*[1] * level, count, *digits[level + 1 :]),
                )
            )
            start += count * stride
            length -= count * stride
        return boxes

    def count_ranges(self, box: Box) -> int:
        """How many separate ranges of the axis a digit box covers.

        One exactly when every digit in front of the last partly taken one
        is a single value; otherwise one per value combination of those
        digits. ``(2, 3)``: ``[0:2, 0:1]`` covers indices 0 and 3 -> 2.
        """
        return math.prod(box.sizes[: self._last_partial_digit(box)])

    def split_into_ranges(self, box: Box) -> list[Box]:
        """Cut a digit box into the boxes that each cover one range.

        The digits in front of the last partly taken one are pinned to each
        of their values in turn; that digit and the ones after it are kept.
        ``(2, 3)``: ``[0:2, 0:1]`` -> ``[0:1, 0:1]`` and ``[1:2, 0:1]``.
        """
        pivot = self._last_partial_digit(box)
        pinned_values = itertools.product(
            *(
                range(offset, offset + size)
                for offset, size in zip(box.offsets[:pivot], box.sizes[:pivot])
            )
        )
        return [
            Box(
                (*values, *box.offsets[pivot:]),
                (*[1] * pivot, *box.sizes[pivot:]),
            )
            for values in pinned_values
        ]

    def range_of(self, box: Box) -> tuple[int, int]:
        """``(start, length)`` of the range a one-range digit box covers."""
        start = 0
        for digit, offset in zip(self.digits, box.offsets):
            start = start * digit + offset
        return start, math.prod(box.sizes)

    def _last_partial_digit(self, box: Box) -> int:
        """The last digit ``box`` does not take whole (0 if it takes all)."""
        i = max(len(self.digits) - 1, 0)
        while i > 0 and box.offsets[i] == 0 and box.sizes[i] == self.digits[i]:
            i -= 1
        return i


def _inner_sizes(shape: Sequence[int]) -> list[int]:
    """Sizes of the inner blocks of a shape, ascending: ``prod(shape[i:])``."""
    return sorted({math.prod(shape[i:]) for i in range(len(shape) + 1)})


def _can_cut_at(size: int, inner_sizes: Sequence[int]) -> bool:
    """Whether a shape can cut its flat index into blocks of ``size``.

    It can when ``size`` lies between two neighbouring inner block sizes, is
    divided by the smaller one and divides the larger one: then it splits the
    axis between the two into two digits.
    """
    for smaller, larger in zip(inner_sizes, inner_sizes[1:]):
        if smaller <= size <= larger:
            return size % smaller == 0 and larger % size == 0
    return False


def _shared_cuts(
    source: Sequence[int], destination: Sequence[int]
) -> list[int]:
    """Where the flat index is cut into the factors two shapes share.

    Every inner block size of either shape that the other shape can cut at
    too, ascending. They are nested: those of one shape are, and a size of one
    shape that the other can cut at lies between two neighbouring sizes of the
    other with divisibility, so it is nested with every size of it. Cutting at
    further common divisors yields more pieces, never fewer.
    """
    a, b = _inner_sizes(source), _inner_sizes(destination)
    return sorted(
        size
        for size in set(a) | set(b)
        if _can_cut_at(size, a) and _can_cut_at(size, b)
    )


@dataclass(frozen=True)
class FactoredShape:
    """A shape written over the factors it shares with another shape.

    ``digits`` are the shape's own digits, most significant first: its flat
    index cut at its inner block sizes and at the shared cuts. Axis ``i`` owns
    the digits ``axis_digits[i]`` and factor ``j`` owns ``factor_digits[j]``,
    both consecutive. ``(6, 4)`` against ``(2, 3, 4)``: digits ``(2, 3, 4)``,
    axes ``0:2`` and ``2:3``, factors ``0:1``, ``1:2`` and ``2:3``. Against
    ``(4, 6)``: digits ``(6, 4)``, axes ``0:1`` and ``1:2``, the one factor
    ``0:2``.

    A *factor box* is a box over the factors -- the coordinates both sides of
    a reshape share. It is a box of this shape exactly when every factor's
    range is one box of that factor's digits and, on every axis, the digits it
    takes cover one range of the axis.
    """

    shape: tuple[int, ...]
    digits: tuple[int, ...]
    axis_digits: tuple[range, ...]
    factor_digits: tuple[range, ...]

    @classmethod
    def over(cls, shape: Sequence[int], cuts: Sequence[int]) -> FactoredShape:
        """``shape`` written over the factors between neighbouring ``cuts``."""
        sizes = sorted(set(_inner_sizes(shape)) | set(cuts), reverse=True)
        # digit k covers the inner block sizes in (sizes[k + 1], sizes[k]]
        position = {size: k for k, size in enumerate(sizes)}

        def owned(blocks):
            return tuple(
                range(position[outer], position[inner])
                for outer, inner in blocks
            )

        cuts = sorted(cuts, reverse=True)
        return cls(
            tuple(shape),
            tuple(outer // inner for outer, inner in zip(sizes, sizes[1:])),
            owned(
                (math.prod(shape[i:]), math.prod(shape[i + 1 :]))
                for i in range(len(shape))
            ),
            owned(zip(cuts, cuts[1:])),
        )

    def to_factor_boxes(
        self, box: Box, max_pieces: int | None = None
    ) -> list[Box]:
        """A box of this shape as factor boxes that are boxes of this shape."""
        digit_boxes = _product(
            [
                radix.boxes_of_range(offset, size)
                for (_, radix), offset, size in zip(
                    self._axes, box.offsets, box.sizes
                )
            ],
            max_pieces,
        )
        return self.split_to_fit(
            [
                factor_box
                for digit_box in digit_boxes
                for factor_box in self._factor_boxes(digit_box, max_pieces)
            ],
            max_pieces,
        )

    def split_to_fit(
        self, factor_boxes: list[Box], max_pieces: int | None = None
    ) -> list[Box]:
        """Cut factor boxes until each one is a box of this shape.

        A box that does not fit is expanded into digit boxes, those are cut
        for the axes and the parts folded back into factor boxes; cutting for
        the axes can leave a factor covering several ranges of its index, so a
        part may need cutting again. The order of the input is kept.
        """
        fitted = []
        pending = list(reversed(factor_boxes))
        while pending:
            factor_box = pending.pop()
            if self.fits(factor_box):
                fitted.append(factor_box)
                if max_pieces is not None and len(fitted) > max_pieces:
                    raise _PieceLimitReached
                continue
            parts = [
                part
                for digit_box in self._digit_boxes(factor_box, max_pieces)
                for cut in self._axis_cuts(digit_box, max_pieces)
                for part in self._factor_boxes(cut, max_pieces)
            ]
            pending.extend(reversed(parts))
        return fitted

    def fits(self, factor_box: Box) -> bool:
        """Whether a factor box is a box of this shape."""
        if self._one_digit_per_factor:
            digit_box = factor_box
        else:
            per_factor = self._per_factor_digit_boxes(factor_box)
            if any(len(boxes) != 1 for boxes in per_factor):
                return False
            digit_box = Box.concat([boxes[0] for boxes in per_factor])
        return all(
            radix.count_ranges(digit_box.select(owned)) == 1
            for owned, radix in self._axes
        )

    def from_factor_box(self, factor_box: Box) -> Box:
        """A factor box that fits this shape, in this shape's coordinates."""
        (digit_box,) = self._digit_boxes(factor_box, None)
        ranges = [
            radix.range_of(digit_box.select(owned))
            for owned, radix in self._axes
        ]
        return Box(
            tuple(start for start, _ in ranges),
            tuple(length for _, length in ranges),
        )

    @functools.cached_property
    def _axes(self) -> tuple[tuple[range, MixedRadixAxis], ...]:
        return tuple((owned, self._radix(owned)) for owned in self.axis_digits)

    @functools.cached_property
    def _factors(self) -> tuple[tuple[range, MixedRadixAxis], ...]:
        return tuple(
            (owned, self._radix(owned)) for owned in self.factor_digits
        )

    @functools.cached_property
    def _one_digit_per_factor(self) -> bool:
        # Then a factor box is its own digit box, as when the two shapes only
        # group axes differently.
        return len(self.digits) == len(self.factor_digits)

    def _radix(self, owned: range) -> MixedRadixAxis:
        return MixedRadixAxis(self.digits[owned.start : owned.stop])

    def _per_factor_digit_boxes(self, factor_box: Box) -> list[list[Box]]:
        return [
            radix.boxes_of_range(offset, size)
            for (_, radix), offset, size in zip(
                self._factors, factor_box.offsets, factor_box.sizes
            )
        ]

    def _digit_boxes(
        self, factor_box: Box, max_pieces: int | None
    ) -> list[Box]:
        """The digit boxes that together cover a factor box."""
        if self._one_digit_per_factor:
            return [factor_box]
        return _product(self._per_factor_digit_boxes(factor_box), max_pieces)

    def _axis_cuts(self, digit_box: Box, max_pieces: int | None) -> list[Box]:
        """Cut a digit box until every axis covers one range."""
        return _product(
            [
                radix.split_into_ranges(digit_box.select(owned))
                for owned, radix in self._axes
            ],
            max_pieces,
        )

    def _factor_boxes(
        self, digit_box: Box, max_pieces: int | None
    ) -> list[Box]:
        """Cut a digit box until every factor covers one range."""
        if self._one_digit_per_factor:
            return [digit_box]
        per_factor = []
        for owned, radix in self._factors:
            ranges = (
                radix.range_of(part)
                for part in radix.split_into_ranges(digit_box.select(owned))
            )
            per_factor.append(
                [Box((start,), (length,)) for start, length in ranges]
            )
        return _product(per_factor, max_pieces)


class Piece(NamedTuple):
    """The same elements, in the same order, as a box of each side."""

    source: Box
    destination: Box


@dataclass(frozen=True)
class ReshapeMapping:
    """A source and a destination shape written over the same factors."""

    source: FactoredShape
    destination: FactoredShape

    @classmethod
    def of(
        cls, source: Sequence[int], destination: Sequence[int]
    ) -> ReshapeMapping:
        """The mapping between two shapes; raises
        :class:`IncompatibleShapesError` when they hold different numbers of
        elements. A plan asks for the same few mappings over and over, so
        they are cached."""
        return cls._of(tuple(source), tuple(destination))

    @classmethod
    @functools.lru_cache(maxsize=1024)
    def _of(
        cls, source: tuple[int, ...], destination: tuple[int, ...]
    ) -> ReshapeMapping:
        if math.prod(source) != math.prod(destination):
            raise IncompatibleShapesError(
                _message(
                    "reshape needs the two shapes to hold the same elements",
                    f"they hold {math.prod(source)} and "
                    f"{math.prod(destination)} elements",
                    source=source,
                    destination=destination,
                )
            )
        cuts = _shared_cuts(source, destination)
        return cls(
            FactoredShape.over(source, cuts),
            FactoredShape.over(destination, cuts),
        )

    def reversed(self) -> ReshapeMapping:
        return ReshapeMapping(self.destination, self.source)

    def carry(self, box: Box) -> Box | None:
        """The destination box of a source box that is a piece already, or
        ``None`` -- the cheap question :meth:`split_into_pieces` answers with
        one piece."""
        try:
            (factor_box,) = self.source.to_factor_boxes(box, max_pieces=1)
        except _PieceLimitReached:
            return None
        if not self.destination.fits(factor_box):
            return None
        return self.destination.from_factor_box(factor_box)

    def split_into_pieces(
        self,
        box: Box,
        max_pieces: int | None = MAX_RESHAPE_PIECES,
        context: dict | None = None,
    ) -> list[Piece]:
        """Cut a box of the source into pieces that are boxes of both sides.

        Converts the box to factor boxes (boxes of the source), cuts them for
        the destination and then the source again until neither cuts any
        more -- a cut for one side can break the other -- and converts each
        result back to both shapes. ``context`` is named in the error raised
        past ``max_pieces``.
        """
        try:
            factor_boxes = self.source.to_factor_boxes(box, max_pieces)
            while True:
                count = len(factor_boxes)
                for side in (self.destination, self.source):
                    factor_boxes = side.split_to_fit(factor_boxes, max_pieces)
                if len(factor_boxes) == count:
                    break
        except _PieceLimitReached:
            raise TooManyPiecesError(
                _message(
                    "reshape needs every region to fall into few boxes of "
                    "both shapes",
                    f"it would take more than {max_pieces} pieces, because "
                    f"no fewer boxes common to {self.source.shape} and "
                    f"{self.destination.shape} make up the region",
                    source=self.source.shape,
                    destination=self.destination.shape,
                    **(context or {}),
                )
            ) from None
        return [
            Piece(
                self.source.from_factor_box(factor_box),
                self.destination.from_factor_box(factor_box),
            )
            for factor_box in factor_boxes
        ]


# ---------------------------------------------------------------------------
# 3. Postprocess chains
# ---------------------------------------------------------------------------


class PostprocessChain:
    """The stages a slice ref's data goes through on its postprocess list.

    Stage 0 is the shape of the source region and every entry gives the next
    stage: a permute reorders the axes, a reshape regroups them, a cast keeps
    them. The entries run on the data itself, so a piece of the ref must be a
    box at every stage, and at every reshape a box of the stages on both sides.
    """

    def __init__(
        self, input_shape: Sequence[int], postprocess_list: list[str] | None
    ):
        self.postprocess_list = postprocess_list
        self.shapes = [tuple(input_shape)]
        # per entry: its permutation, its ReshapeMapping, or None for a cast
        self._steps = []
        for operation in postprocess_list or []:
            shape = self.shapes[-1]
            if is_permute(operation):
                step = tuple(ast.literal_eval(operation))
                shape = tuple(transpose_list(shape, step))
            elif is_reshape(operation):
                step = ReshapeMapping.of(shape, reshape_op_shape(operation))
                shape = step.destination.shape
            else:
                step = None
            self._steps.append(step)
            self.shapes.append(shape)

    def pieces(
        self,
        box: Box,
        max_pieces: int | None = MAX_RESHAPE_PIECES,
        context: dict | None = None,
    ) -> list[list[Box]]:
        """Cut a box of the last stage into pieces that are boxes of every
        stage; each piece as its box at every stage.

        Pulls the box back to stage 0 -- each reshape on the way cuts it into
        its own pieces -- and pushes every stage-0 box forward: a reshape it is
        not a piece of cuts it there, and the parts are pulled back and pushed
        again. Each reshape cuts as coarsely as it can and a permute keeps
        boxes, so this ends at the coarsest cut every stage accepts (the
        tests compare it with an exhaustive search on small chains).
        """
        pending = list(
            reversed(self._pull(len(self._steps), box, max_pieces, context))
        )
        pieces = []
        while pending:
            stages, cut = self._push(pending.pop(), max_pieces, context)
            if cut is None:
                pieces.append(stages)
                if max_pieces is not None and len(pieces) > max_pieces:
                    raise TooManyPiecesError(
                        _message(
                            "reshape needs every region to fall into few "
                            "boxes of every stage of its postprocess list",
                            f"it would take more than {max_pieces} pieces",
                            stages=self.shapes,
                            postprocess_list=self.postprocess_list,
                            **(context or {}),
                        )
                    )
                continue
            index, parts = cut
            pending.extend(
                reversed(
                    [
                        pulled
                        for part in parts
                        for pulled in self._pull(
                            index, part.source, max_pieces, context
                        )
                    ]
                )
            )
        return pieces

    def localized(self, stages: Sequence[Box]) -> list[str] | None:
        """The postprocess list of one piece: every reshape entry names the
        piece's own shape after it."""
        if self.postprocess_list is None:
            return None
        return [
            make_reshape_op(stages[index + 1].sizes)
            if isinstance(step, ReshapeMapping)
            else operation
            for index, (operation, step) in enumerate(
                zip(self.postprocess_list, self._steps)
            )
        ]

    def _pull(
        self,
        stage: int,
        box: Box,
        max_pieces: int | None,
        context: dict | None,
    ) -> list[Box]:
        """Stage-0 boxes that together become ``box`` at ``stage``."""
        boxes = [box]
        for step in reversed(self._steps[:stage]):
            if isinstance(step, ReshapeMapping):
                backward = step.reversed()
                boxes = [
                    piece.destination
                    for b in boxes
                    for piece in backward.split_into_pieces(
                        b, max_pieces, context
                    )
                ]
            elif step is not None:
                inverse = invert_permutation(step)
                boxes = [b.permuted(inverse) for b in boxes]
        return boxes

    def _push(
        self, box: Box, max_pieces: int | None, context: dict | None
    ) -> tuple[list[Box], tuple[int, list[Piece]] | None]:
        """``box`` at every stage, or, at the first reshape it is not a piece
        of, the stages so far and that reshape's pieces of it."""
        stages = [box]
        for index, step in enumerate(self._steps):
            if isinstance(step, ReshapeMapping):
                carried = step.carry(stages[-1])
                if carried is None:
                    parts = step.split_into_pieces(
                        stages[-1], max_pieces, context
                    )
                    return stages, (index, parts)
                stages.append(carried)
            elif step is not None:
                stages.append(stages[-1].permuted(step))
            else:
                stages.append(stages[-1])
        return stages, None


# ---------------------------------------------------------------------------
# 4. Slice refs
#
# A slice ref ``(key, src_slice, dst_slice, postprocess_list)`` puts the
# ``src_slice`` region of ``key``, once its postprocess list has run, at
# ``dst_slice``. A reshape entry in that list names the local shape the data
# takes, so the list is a chain over the source region's own shape. Everything
# below works relative to the ref's own regions, which is what keeps it right
# wherever later statements move the ref.
# ---------------------------------------------------------------------------
def resolve_reshape_target(
    attr_shape: tuple[int, ...],
    source_shape: tuple[int, ...],
    dst_metadata_shape: tuple[int, ...] | None,
    reverse: bool,
    written_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """The shape a reshape statement produces.

    ``reshape=`` names the shape of the statement's right-hand side, like
    ``paddle.reshape``. ``dst_metadata_shape`` is the destination's global
    shape from its shard info (``None`` for an intermediate variable or on the
    save path). With ``aoa_config_reverse`` the statement runs right to left,
    so the attribute names the shape going *in*; the one coming out is the
    left-hand side's: its shard info, or else ``written_shape``, the shape the
    statements give it when run as written. The two must agree when both are
    known.
    """
    if not reverse:
        if dst_metadata_shape is not None and dst_metadata_shape != attr_shape:
            raise ReshapeAttributeError(
                _message(
                    f"reshape={list(attr_shape)} must name the shape the "
                    "statement produces",
                    f"the destination is {dst_metadata_shape}. "
                    f"Write reshape='{list(dst_metadata_shape)}'",
                    source=source_shape,
                    destination=dst_metadata_shape,
                )
            )
        return attr_shape
    if attr_shape != source_shape:
        raise ReshapeAttributeError(
            _message(
                "with aoa_config_reverse the statement runs right to left, so "
                "reshape= names the shape it reads",
                f"it reads {source_shape}",
                attribute=list(attr_shape),
                source=source_shape,
            )
        )
    if (
        dst_metadata_shape is not None
        and written_shape is not None
        and dst_metadata_shape != written_shape
    ):
        raise ReshapeAttributeError(
            _message(
                "with aoa_config_reverse a reshape writes the left-hand side, "
                "whose shard info and the statements must agree on its shape",
                f"the shard info says {dst_metadata_shape}, the statements "
                f"give {written_shape} when run as written",
                source=source_shape,
            )
        )
    destination = (
        dst_metadata_shape if dst_metadata_shape is not None else written_shape
    )
    if destination is None:
        raise ReshapeAttributeError(
            _message(
                "with aoa_config_reverse the shape a reshape writes comes from "
                "the destination's shard info, or for an intermediate variable "
                "from the statements run as written",
                "this destination has neither",
                source=source_shape,
            )
        )
    return destination


def reshape_slice_refs(
    slice_refs: list[SliceRef],
    mapping: ReshapeMapping,
    max_pieces: int | None = MAX_RESHAPE_PIECES,
) -> list[SliceRef]:
    """The slice refs of the reshaped tensor.

    Each input ref's destination region (a box of ``mapping.source``) is cut
    into pieces; each piece is restricted to the ref and, reading exactly its
    part of the source, ends its postprocess list with a reshape to the
    piece's shape. When the ref reshapes already, restricting can cut a piece
    again; each part is then cut for ``mapping`` anew, until every part is one
    ref. ``max_pieces`` bounds how finely a region may be cut before it is
    refused (see ``MAX_RESHAPE_PIECES``).
    """

    def pieces(ref: SliceRef) -> list[tuple[SliceRef, Piece]]:
        return [
            (ref, piece)
            for piece in mapping.split_into_pieces(
                Box.from_slices(ref[2]),
                max_pieces=max_pieces,
                context={"source_slice": ref[2]},
            )
        ]

    reshaped = []
    pending = list(
        reversed([item for ref in slice_refs for item in pieces(ref)])
    )
    while pending:
        ref, piece = pending.pop()
        restricted = restrict_slice_ref(
            ref, piece.source, max_pieces=max_pieces
        )
        if len(restricted) > 1:
            pending.extend(
                reversed([item for part in restricted for item in pieces(part)])
            )
            continue
        ((key, src_slice, _, postprocess_list),) = restricted
        reshaped.append(
            (
                key,
                src_slice,
                piece.destination.to_slices(),
                [
                    *(postprocess_list or []),
                    make_reshape_op(piece.destination.sizes),
                ],
            )
        )
    return reshaped


def restrict_slice_ref(
    ref: SliceRef,
    region: Box,
    max_pieces: int | None = MAX_RESHAPE_PIECES,
    **context: object,
) -> list[SliceRef]:
    """The refs that put exactly ``region`` of ``ref``'s destination in place.

    ``region`` lies inside the ref's destination slice, in the same
    coordinates. Without a reshape in the postprocess list this is one ref:
    permutes and casts keep relative positions. ``max_pieces`` bounds how
    finely the region may be cut (see ``MAX_RESHAPE_PIECES``). ``context`` is
    named in the error raised for a region that falls into too many pieces.
    """
    key, src_slice, dst_slice, postprocess_list = ref
    source_box = Box.from_slices(src_slice)
    destination_box = Box.from_slices(dst_slice)
    chain = PostprocessChain(source_box.sizes, postprocess_list)
    return [
        (
            key,
            stages[0].offset_by(source_box).to_slices(),
            stages[-1].offset_by(destination_box).to_slices(),
            chain.localized(stages),
        )
        for stages in chain.pieces(
            region.relative_to(destination_box),
            max_pieces=max_pieces,
            context=context,
        )
    ]
