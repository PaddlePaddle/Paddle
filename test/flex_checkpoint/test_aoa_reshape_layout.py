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
"""The integer geometry behind the AoA ``reshape`` primitive.

The geometry and chain sections of ``aoa/reshape.py`` import nothing from
Paddle, so they are checked here directly: against a numpy oracle, and against
the fewest pieces an exhaustive search finds, on every small case.
"""

import ast
import functools
import itertools
import math
import unittest

import numpy as np

from paddle.distributed.flex_checkpoint.aoa.postprocess import (
    is_permute,
    is_reshape,
    make_reshape_op,
    reshape_op_shape,
)
from paddle.distributed.flex_checkpoint.aoa.reshape import (
    Box,
    IncompatibleShapesError,
    MixedRadixAxis,
    Piece,
    PostprocessChain,
    ReshapeMapping,
    TooManyPiecesError,
)


def _shapes(count, max_rank=3):
    """Every shape of up to ``max_rank`` axes, none of size one, holding
    ``count`` elements."""
    shapes = []
    for rank in range(1, max_rank + 1):
        for dims in itertools.product(range(2, count + 1), repeat=rank):
            if math.prod(dims) == count:
                shapes.append(dims)
    return shapes


def _boxes(shape):
    ranges = [
        [
            (start, stop - start)
            for start in range(dim)
            for stop in range(start + 1, dim + 1)
        ]
        for dim in shape
    ]
    for per_axis in itertools.product(*ranges):
        yield Box(tuple(o for o, _ in per_axis), tuple(s for _, s in per_axis))


def _region(array, box):
    return array[box.to_slices()]


def _run(array, postprocess_list):
    """Apply a postprocess list to an array, as the copy layer does."""
    for operation in postprocess_list:
        if is_permute(operation):
            array = np.transpose(array, ast.literal_eval(operation))
        elif is_reshape(operation):
            array = array.reshape(reshape_op_shape(operation))
    return array


def _stage_coords(shape, postprocess_list):
    """The coordinates every element (by its stage-0 flat index) has at
    every stage of a postprocess list."""
    coords = np.array(np.unravel_index(np.arange(math.prod(shape)), shape)).T
    stages = [coords]
    for operation in postprocess_list:
        if is_permute(operation):
            permutation = ast.literal_eval(operation)
            coords = coords[:, permutation]
            shape = [shape[p] for p in permutation]
        elif is_reshape(operation):
            flat = np.ravel_multi_index(coords.T, shape)
            shape = reshape_op_shape(operation)
            coords = np.array(np.unravel_index(flat, shape)).T
        stages.append(coords)
    return stages


def _fewest_pieces(shape, postprocess_list, region, region_stage):
    """The fewest stage-0 boxes that are boxes at every stage and together
    hold exactly the elements ``region`` covers at ``region_stage``."""
    stages = _stage_coords(shape, postprocess_list)
    low = np.array(region.offsets)
    inside = np.all(
        (stages[region_stage] >= low)
        & (stages[region_stage] < low + np.array(region.sizes)),
        axis=1,
    )
    elements = np.flatnonzero(inside)
    bit = {element: 1 << i for i, element in enumerate(elements)}
    starting_at = {}  # lowest element bit -> candidate masks
    for box in _boxes(shape):
        grid = np.indices(box.sizes).reshape(len(shape), -1)
        ids = np.ravel_multi_index(grid + np.array(box.offsets)[:, None], shape)
        if not inside[ids].all() or any(
            np.prod(c[ids].max(0) - c[ids].min(0) + 1) != len(ids)
            for c in stages[1:]
        ):
            continue
        mask = sum(bit[i] for i in ids)
        starting_at.setdefault(mask & -mask, []).append(mask)
    full = (1 << len(elements)) - 1

    @functools.cache
    def fewest(covered):
        if covered == full:
            return 0
        free = ~covered & full
        return min(
            1 + fewest(covered | mask)
            for mask in starting_at.get(free & -free, ())
            if not mask & covered
        )

    return fewest(0)


class TestBoxFromSlices(unittest.TestCase):
    def test_integer_bounds(self):
        self.assertEqual(
            Box.from_slices((slice(1, 3), slice(0, 4, 1))),
            Box((1, 0), (2, 4)),
        )
        # numpy integers (e.g. from shard metadata) are integers too
        self.assertEqual(
            Box.from_slices((slice(np.int64(1), np.int64(3)),)),
            Box((1,), (2,)),
        )

    def test_open_bounds_and_strides_are_refused(self):
        for bad in (slice(None, 3), slice(0, None), slice(0, 4, 2)):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(AssertionError, "integer bounds"),
            ):
                Box.from_slices((slice(0, 2), bad))


class TestMixedRadixAxis(unittest.TestCase):
    """An axis of 6 made of the digits (2, 3): index = e * 3 + h."""

    AXIS = MixedRadixAxis((2, 3))

    def test_boxes_of_range(self):
        cases = {
            (0, 6): [Box((0, 0), (2, 3))],
            (2, 4): [Box((0, 2), (1, 1)), Box((1, 0), (1, 3))],
            (1, 4): [Box((0, 1), (1, 2)), Box((1, 0), (1, 2))],
        }
        for (start, length), expected in cases.items():
            with self.subTest(start=start, length=length):
                self.assertEqual(
                    self.AXIS.boxes_of_range(start, length), expected
                )

    def test_count_and_split_ranges(self):
        cases = {
            Box((0, 0), (2, 3)): [Box((0, 0), (2, 3))],  # indices 0-5
            Box((1, 1), (1, 2)): [Box((1, 1), (1, 2))],  # indices 4-5
            Box((0, 0), (2, 1)): [Box((0, 0), (1, 1)), Box((1, 0), (1, 1))],
        }
        for box, expected in cases.items():
            with self.subTest(box=box):
                self.assertEqual(self.AXIS.count_ranges(box), len(expected))
                self.assertEqual(self.AXIS.split_into_ranges(box), expected)

    def test_range_of(self):
        self.assertEqual(self.AXIS.range_of(Box((1, 0), (1, 3))), (3, 3))

    def test_size_one_axis_has_no_digits(self):
        axis = MixedRadixAxis(())
        self.assertEqual(axis.boxes_of_range(0, 1), [Box((), ())])
        self.assertEqual(axis.count_ranges(Box((), ())), 1)
        self.assertEqual(axis.range_of(Box((), ())), (0, 1))


class TestFactoredShape(unittest.TestCase):
    def _assert_layout(self, shape, other, digits, axis_digits, factor_digits):
        mapping = ReshapeMapping.of(shape, other)
        self.assertEqual(mapping.source.digits, digits)
        self.assertEqual(mapping.source.axis_digits, axis_digits)
        self.assertEqual(mapping.source.factor_digits, factor_digits)

    def test_grouped_gemm_layouts(self):
        # (6, 4) groups the factors of (2, 3, 2, 2): one digit per factor.
        self._assert_layout(
            (6, 4),
            (2, 3, 2, 2),
            (2, 3, 2, 2),
            (range(0, 2), range(2, 4)),
            (range(0, 1), range(1, 2), range(2, 3), range(3, 4)),
        )

    def test_factors_finer_than_both_shapes(self):
        self._assert_layout(
            (2, 6),
            (4, 3),
            (2, 2, 3),
            (range(0, 1), range(1, 3)),
            (range(0, 1), range(1, 2), range(2, 3)),
        )

    def test_size_one_axes_own_no_digits(self):
        self._assert_layout(
            (1, 6),
            (2, 1, 3),
            (2, 3),
            (range(0), range(0, 2)),
            (range(0, 1), range(1, 2)),
        )

    def test_a_factor_the_shapes_write_differently(self):
        # No axis boundary of (6, 4) is one of (4, 6): the whole index is one
        # factor of 24, written 6 x 4 by one shape and 4 x 6 by the other.
        self._assert_layout(
            (6, 4), (4, 6), (6, 4), (range(0, 1), range(1, 2)), (range(0, 2),)
        )
        self._assert_layout(
            (4, 6), (6, 4), (4, 6), (range(0, 1), range(1, 2)), (range(0, 2),)
        )
        # The leading 2 is still shared; only the trailing 24 is written
        # differently.
        self._assert_layout(
            (2, 6, 4),
            (2, 4, 6),
            (2, 6, 4),
            (range(0, 1), range(1, 2), range(2, 3)),
            (range(0, 1), range(1, 3)),
        )

    def test_element_counts_must_match(self):
        with self.assertRaisesRegex(
            IncompatibleShapesError, "they hold 20 and 24 elements"
        ):
            ReshapeMapping.of((5, 4), (24,))

    def test_reversed_swaps_the_sides(self):
        for source, destination in (
            ((6, 4), (2, 3, 4)),
            ((1, 6), (2, 1, 3)),
            ((6, 4), (4, 6)),
        ):
            self.assertEqual(
                ReshapeMapping.of(source, destination).reversed(),
                ReshapeMapping.of(destination, source),
            )


class TestSplitIntoPieces(unittest.TestCase):
    """Every box of every small shape pair, against numpy and against the
    fewest pieces there are."""

    def _assert_exact(self, mapping, box):
        pieces = mapping.split_into_pieces(box, max_pieces=None)
        source, destination = mapping.source.shape, mapping.destination.shape
        count = math.prod(source)
        src = np.arange(count).reshape(source)
        dst = np.arange(count).reshape(destination)
        covered = np.zeros(source, dtype=int)
        for piece in pieces:
            # same elements in the same order on both sides
            np.testing.assert_array_equal(
                _region(src, piece.source).reshape(piece.destination.sizes),
                _region(dst, piece.destination),
            )
            _region(covered, piece.source)[...] += 1
        expected = np.zeros(source, dtype=int)
        _region(expected, box)[...] = 1
        np.testing.assert_array_equal(covered, expected)
        return pieces

    def test_every_box_of_every_small_shape_pair(self):
        checked = 0
        for count in (4, 6, 8, 12):
            for source, destination in itertools.permutations(
                _shapes(count), 2
            ):
                mapping = ReshapeMapping.of(source, destination)
                reshape = [make_reshape_op(destination)]
                for box in _boxes(source):
                    pieces = self._assert_exact(mapping, box)
                    self.assertEqual(
                        len(pieces),
                        _fewest_pieces(source, reshape, box, 0),
                        f"{source} -> {destination}, {box}",
                    )
                    checked += 1
        self.assertGreater(checked, 3000)

    def test_a_box_of_both_shapes_is_one_piece(self):
        # [E, H, I] cut along I: six separate stretches of the flat order of
        # [E * H, I], yet a box of both shapes, so a single piece.
        pieces = ReshapeMapping.of((2, 3, 4), (6, 4)).split_into_pieces(
            Box((0, 0, 0), (2, 3, 2))
        )
        self.assertEqual(
            pieces, [Piece(Box((0, 0, 0), (2, 3, 2)), Box((0, 0), (6, 2)))]
        )

    def test_shapes_that_write_a_factor_differently(self):
        mapping = ReshapeMapping.of((6, 4), (4, 6))
        # the top half is rows 0-2 of one and rows 0-1 of the other
        self.assertEqual(
            mapping.split_into_pieces(Box((0, 0), (3, 4))),
            [Piece(Box((0, 0), (3, 4)), Box((0, 0), (2, 6)))],
        )
        # the left half falls apart into its rows
        self.assertEqual(len(mapping.split_into_pieces(Box((0, 0), (6, 2)))), 6)

    def test_limit_refuses_instead_of_enumerating(self):
        # one index of the last axis of [4, 4, 4], flattened: every element of
        # it is its own stretch of the flat order, so 16 pieces
        mapping = ReshapeMapping.of((4, 4, 4), (64,))
        box = Box((0, 0, 0), (4, 4, 1))
        self.assertEqual(len(mapping.split_into_pieces(box)), 16)
        with self.assertRaisesRegex(TooManyPiecesError, "more than 15 pieces"):
            mapping.split_into_pieces(box, max_pieces=15)


class TestPostprocessChain(unittest.TestCase):
    """Several reshapes in one postprocess list, with permutes between them."""

    def _assert_exact(self, shape, postprocess_list, box):
        chain = PostprocessChain(shape, postprocess_list)
        source = np.arange(math.prod(shape)).reshape(shape)
        result = _run(source, postprocess_list)
        covered = np.zeros(result.shape, dtype=int)
        pieces = chain.pieces(box, max_pieces=None)
        for stages in pieces:
            np.testing.assert_array_equal(
                _run(_region(source, stages[0]), chain.localized(stages)),
                _region(result, stages[-1]),
            )
            _region(covered, stages[-1])[...] += 1
        expected = np.zeros(result.shape, dtype=int)
        _region(expected, box)[...] = 1
        np.testing.assert_array_equal(covered, expected)
        return pieces

    def _chains(self, count):
        """reshape -> reshape and reshape -> permute -> reshape, every way."""
        for shape, middle, last in itertools.product(_shapes(count), repeat=3):
            if middle == shape:
                continue
            yield shape, [make_reshape_op(middle), make_reshape_op(last)]
            for permutation in itertools.permutations(range(len(middle))):
                if list(permutation) != sorted(permutation):
                    yield (
                        shape,
                        [
                            make_reshape_op(middle),
                            str(list(permutation)),
                            make_reshape_op(last),
                        ],
                    )

    def test_every_box_through_every_small_chain(self):
        checked = 0
        for count in (6, 8):
            for shape, postprocess_list in self._chains(count):
                last = reshape_op_shape(postprocess_list[-1])
                for box in _boxes(last):
                    pieces = self._assert_exact(shape, postprocess_list, box)
                    self.assertEqual(
                        len(pieces),
                        _fewest_pieces(
                            shape, postprocess_list, box, len(postprocess_list)
                        ),
                        f"{shape} {postprocess_list}, {box}",
                    )
                    checked += 1
        self.assertGreater(checked, 1000)

    def test_two_reshapes_that_no_shared_atom_explains(self):
        # (2, 3) transposed is (3, 2); flattened and regrouped as (2, 3) that
        # is a reshape (3, 2) -> (2, 3), whose only shared factor is the whole
        # index. The whole tensor is still one piece.
        postprocess_list = ["[1, 0]", "reshape:[6]", "reshape:[2, 3]"]
        pieces = self._assert_exact(
            (2, 3), postprocess_list, Box((0, 0), (2, 3))
        )
        self.assertEqual(len(pieces), 1)

    def test_permutes_and_casts_keep_one_piece(self):
        chain = PostprocessChain((2, 3, 4), ["[2, 0, 1]", "float16"])
        ((source, permuted, cast),) = chain.pieces(Box((1, 0, 1), (2, 1, 2)))
        self.assertEqual(source, Box((0, 1, 1), (1, 2, 2)))
        self.assertEqual(permuted, Box((1, 0, 1), (2, 1, 2)))
        self.assertEqual(cast, permuted)
        self.assertEqual(
            chain.localized((source, permuted, cast)), ["[2, 0, 1]", "float16"]
        )
        self.assertIsNone(
            PostprocessChain((2, 3), None).localized([Box((0, 0), (2, 3))])
        )

    def test_each_piece_names_its_own_shapes(self):
        # (6, 4) -> (2, 3, 4) -> permuted to (2, 4, 3) -> (8, 3); the rows of
        # the last stage that hold expert 1 are one piece of every stage.
        postprocess_list = ["reshape:[2, 3, 4]", "[0, 2, 1]", "reshape:[8, 3]"]
        chain = PostprocessChain((6, 4), postprocess_list)
        (stages,) = chain.pieces(Box((4, 0), (4, 3)))
        self.assertEqual(stages[0], Box((3, 0), (3, 4)))
        self.assertEqual(
            chain.localized(stages),
            ["reshape:[1, 3, 4]", "[0, 2, 1]", "reshape:[4, 3]"],
        )


if __name__ == '__main__':
    unittest.main()
