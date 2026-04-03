# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from test_parallel_dygraph_dataparallel import TestMultipleAccelerators

import paddle
from paddle.distributed.fleet.meta_parallel.pipeline_parallel import (
    _can_free,
    _collect_all_tensors,
    _release_input,
    _release_output,
)


class TestPipelineParallel(TestMultipleAccelerators):
    def test_pipeline_parallel(self):
        self.run_mnist_2accelerators('hybrid_parallel_pp_alexnet.py')


class TestModelParallelWithRecompute(TestMultipleAccelerators):
    def test_model_parallel_with_recompute(self):
        self.run_mnist_2accelerators("dygraph_recompute_hybrid.py")


class TestCanFree(unittest.TestCase):
    """Unit tests for _can_free(), covering all branches."""

    def test_none_is_not_freeable(self):
        self.assertFalse(_can_free(None))

    def test_non_tensor_is_not_freeable(self):
        self.assertFalse(_can_free(42))
        self.assertFalse(_can_free("string"))
        self.assertFalse(_can_free([1, 2, 3]))

    def test_initialized_tensor_is_freeable(self):
        t = paddle.rand([3, 4])
        self.assertTrue(_can_free(t))

    def test_uninitialized_tensor_is_not_freeable(self):
        t = paddle.rand([3, 4])
        t._clear_dataptr()
        self.assertFalse(_can_free(t))

    def test_pp_can_free_flag_overrides(self):
        # Tensor marked pp_can_free=True should always be freeable
        # (the flag is intended to override the inplace_version check)
        t = paddle.rand([3, 4])
        t.pp_can_free = True
        self.assertTrue(_can_free(t))


class TestCollectAllTensors(unittest.TestCase):
    """Unit tests for _collect_all_tensors(), covering all branches."""

    def test_single_tensor(self):
        t = paddle.rand([2, 3])
        result = set()
        _collect_all_tensors(t, result)
        self.assertIn(t, result)
        self.assertEqual(len(result), 1)

    def test_tuple_of_tensors(self):
        t1, t2 = paddle.rand([2, 3]), paddle.rand([2, 3])
        result = set()
        _collect_all_tensors((t1, t2), result)
        self.assertIn(t1, result)
        self.assertIn(t2, result)

    def test_list_of_tensors(self):
        t1, t2 = paddle.rand([2, 3]), paddle.rand([2, 3])
        result = set()
        _collect_all_tensors([t1, t2], result)
        self.assertIn(t1, result)
        self.assertIn(t2, result)

    def test_dict_of_tensors(self):
        t1, t2 = paddle.rand([2, 3]), paddle.rand([2, 3])
        result = set()
        _collect_all_tensors({'a': t1, 'b': t2}, result)
        self.assertIn(t1, result)
        self.assertIn(t2, result)

    def test_nested_structure(self):
        t1 = paddle.rand([2, 3])
        t2 = paddle.rand([2, 3])
        t3 = paddle.rand([2, 3])
        result = set()
        _collect_all_tensors(((t1, t2), [t3]), result)
        self.assertEqual(result, {t1, t2, t3})

    def test_duplicate_tensor_collected_once(self):
        t = paddle.rand([2, 3])
        result = set()
        _collect_all_tensors((t, t), result)
        self.assertEqual(len(result), 1)
        self.assertIn(t, result)

    def test_duplicate_tensor_already_in_set_triggers_debug_log(self):
        # Pre-populate tensor_set with t, then feed t again via _collect_all_tensors.
        # This makes `current in tensor_set` True, triggering the logger.debug branch.
        t = paddle.rand([2, 3])
        result = {t}
        _collect_all_tensors(t, result)
        self.assertEqual(len(result), 1)  # still just one tensor

    def test_non_tensor_elements_ignored(self):
        t = paddle.rand([2, 3])
        result = set()
        _collect_all_tensors((t, 42, None, "string"), result)
        self.assertEqual(result, {t})

    def test_empty_structures(self):
        result = set()
        _collect_all_tensors((), result)
        _collect_all_tensors([], result)
        _collect_all_tensors({}, result)
        self.assertEqual(len(result), 0)


class TestReleaseOutput(unittest.TestCase):
    """Unit tests for _release_output(), covering all branches."""

    def test_single_freeable_tensor_is_cleared(self):
        t = paddle.rand([3, 4])
        self.assertTrue(t._is_initialized())
        _release_output(t)
        self.assertFalse(t._is_initialized())

    def test_tuple_of_freeable_tensors_all_cleared(self):
        t1, t2 = paddle.rand([3, 4]), paddle.rand([3, 4])
        _release_output((t1, t2))
        self.assertFalse(t1._is_initialized())
        self.assertFalse(t2._is_initialized())

    def test_list_of_freeable_tensors_all_cleared(self):
        t1, t2 = paddle.rand([3, 4]), paddle.rand([3, 4])
        _release_output([t1, t2])
        self.assertFalse(t1._is_initialized())
        self.assertFalse(t2._is_initialized())

    def test_uninitialized_tensor_not_cleared_again(self):
        t = paddle.rand([3, 4])
        t._clear_dataptr()
        # Should not raise, just skip non-freeable tensors
        _release_output(t)
        self.assertFalse(t._is_initialized())

    def test_none_input_does_not_raise(self):
        _release_output(None)

    def test_dict_tensors_cleared(self):
        t1, t2 = paddle.rand([2, 3]), paddle.rand([2, 3])
        _release_output({'k1': t1, 'k2': t2})
        self.assertFalse(t1._is_initialized())
        self.assertFalse(t2._is_initialized())


class TestReleaseInput(unittest.TestCase):
    """Unit tests for _release_input(), covering all branches."""

    def test_input_not_in_output_is_cleared(self):
        inp = paddle.rand([3, 4])
        out = paddle.rand([3, 4])  # different object
        _release_input(inp, out)
        self.assertFalse(inp._is_initialized())

    def test_input_same_as_output_is_not_cleared(self):
        # Residual connection: same tensor object in both input and output
        t = paddle.rand([3, 4])
        _release_input(t, t)
        # t should NOT be freed because it appears in output
        self.assertTrue(t._is_initialized())

    def test_tuple_input_partial_release(self):
        shared = paddle.rand([3, 4])
        independent = paddle.rand([3, 4])
        out = (
            shared,
            paddle.rand([3, 4]),
        )  # shared is in output, independent is not
        _release_input((shared, independent), out)
        self.assertTrue(
            shared._is_initialized()
        )  # protected: appears in output
        self.assertFalse(independent._is_initialized())  # freed: not in output

    def test_non_freeable_input_not_cleared(self):
        inp = paddle.rand([3, 4])
        inp._clear_dataptr()  # already not initialized
        out = paddle.rand([3, 4])
        _release_input(inp, out)  # should not raise
        self.assertFalse(inp._is_initialized())

    def test_none_output_releases_all_inputs(self):
        inp = paddle.rand([3, 4])
        _release_input(inp, None)
        self.assertFalse(inp._is_initialized())


if __name__ == "__main__":
    unittest.main()
