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

import unittest

import paddle
from paddle.distributed.fleet.meta_parallel.pp_utils.forward_backward_overlap_utils import (
    ScheduleNode,
    clone_and_clear_dataptr,
    detach_and_requires_grad,
)


def simple_forward_func(inputs):
    return inputs * 2


def forward_func_with_labels(inputs, labels):
    return (inputs * 2, labels * 3)


class TestScheduleNode(unittest.TestCase):
    def test_init(self):
        """Test ScheduleNode initialization."""
        node = ScheduleNode(simple_forward_func, name="test_node")
        self.assertEqual(node.name, "test_node")
        self.assertEqual(node.fwd_func, simple_forward_func)
        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)

    def test_init_default_name(self):
        """Test ScheduleNode initialization with default name."""
        node = ScheduleNode(simple_forward_func)
        self.assertEqual(node.name, "")

    def test_forward_basic(self):
        """Test forward pass with basic inputs."""
        node = ScheduleNode(simple_forward_func)
        inputs = paddle.randn([2, 3])
        outputs = node.forward(inputs)

        self.assertIsNotNone(node.inputs)
        self.assertIsNotNone(node.outputs)
        self.assertEqual(outputs.shape, inputs.shape)
        self.assertTrue(paddle.allclose(outputs, inputs * 2))

    def test_forward_with_tuple_inputs(self):
        """Test forward pass with tuple inputs."""
        node = ScheduleNode(lambda inputs: (inputs[0] * 2, inputs[1] * 3))
        input1 = paddle.randn([2, 3])
        input2 = paddle.randn([2, 3])
        outputs = node.forward((input1, input2))

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(len(outputs), 2)

    def test_forward_with_kwargs(self):
        """Test forward pass with keyword arguments."""
        node = ScheduleNode(lambda x, factor=1: x * factor)
        inputs = paddle.randn([2, 3])
        outputs = node.forward(inputs, factor=3)

        self.assertTrue(paddle.allclose(outputs, inputs * 3))

    def test_forward_with_labels(self):
        """Test forward pass with labels set."""
        node = ScheduleNode(forward_func_with_labels, name="test_node")
        node.labels = paddle.ones([2, 3])
        inputs = paddle.randn([2, 3])
        outputs = node.forward(inputs)

        self.assertIsNotNone(node.outputs)

    def test_forward_with_scale_loss_factor(self):
        """Test forward pass with scale_loss_factor."""
        node = ScheduleNode(simple_forward_func)
        node.scale_loss_factor = 2.0
        inputs = paddle.randn([2, 3])
        outputs = node.forward(inputs)

        # Output should be scaled by 1/scale_loss_factor
        self.assertTrue(paddle.allclose(outputs, inputs))

    def test_backward_basic(self):
        """Test backward pass."""
        node = ScheduleNode(simple_forward_func)
        inputs = paddle.randn([2, 3])
        inputs.stop_gradient = False
        outputs = node.forward(inputs)
        grads = node.backward()

        # Check that gradients are returned
        self.assertIsInstance(grads, tuple)

    def test_backward_with_scaler(self):
        """Test backward pass with scaler."""
        node = ScheduleNode(simple_forward_func)
        inputs = paddle.randn([2, 3])
        inputs.stop_gradient = False
        node.forward(inputs)

        scaler = paddle.amp.GradScaler(init_loss_scaling=2.0)
        grads = node.backward(scaler=scaler)

        self.assertIsInstance(grads, tuple)

    def test_backward_with_output_grad(self):
        """Test backward pass with provided output gradients."""
        node = ScheduleNode(simple_forward_func)
        inputs = paddle.randn([2, 3])
        inputs.stop_gradient = False
        node.forward(inputs)

        output_grad = paddle.ones([2, 3])
        grads = node.backward(output_grad=output_grad)

        self.assertIsInstance(grads, tuple)

    def test_reset_states(self):
        """Test _reset_states method."""
        node = ScheduleNode(simple_forward_func)
        node.labels = paddle.ones([2, 3])
        node.scale_loss_factor = 2.0

        node._reset_states()

        self.assertIsNone(node.inputs)
        self.assertIsNone(node.outputs)
        self.assertIsNone(node.labels)
        self.assertIsNone(node.scale_loss_factor)


class TestDetachAndRequiresGrad(unittest.TestCase):
    def test_detach_single_tensor(self):
        """Test detach_and_requires_grad with a single tensor."""
        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = False
        result = detach_and_requires_grad(tensor)

        self.assertIsInstance(result, paddle.Tensor)
        self.assertFalse(result.stop_gradient)

    def test_detach_tensor_with_stop_gradient_true(self):
        """Test detach_and_requires_grad with stop_gradient=True."""
        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = True
        result = detach_and_requires_grad(tensor)

        self.assertTrue(result.stop_gradient)

    def test_detach_list_of_tensors(self):
        """Test detach_and_requires_grad with a list of tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor1.stop_gradient = False
        tensor2 = paddle.randn([2, 3])
        tensor2.stop_gradient = True
        inputs = [tensor1, tensor2]
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].stop_gradient)
        self.assertTrue(result[1].stop_gradient)

    def test_detach_tuple_of_tensors(self):
        """Test detach_and_requires_grad with a tuple of tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor1.stop_gradient = False
        tensor2 = paddle.randn([2, 3])
        tensor2.stop_gradient = False
        inputs = (tensor1, tensor2)
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0].stop_gradient)
        self.assertFalse(result[1].stop_gradient)

    def test_detach_nested_tuple(self):
        """Test detach_and_requires_grad with nested tuple."""
        tensor1 = paddle.randn([2, 3])
        tensor1.stop_gradient = False
        tensor2 = paddle.randn([2, 3])
        tensor2.stop_gradient = True
        inputs = ((tensor1,), (tensor2,))
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, tuple)
        self.assertIsInstance(result[0], tuple)
        self.assertFalse(result[0][0].stop_gradient)
        self.assertTrue(result[1][0].stop_gradient)

    def test_detach_nested_list(self):
        """Test detach_and_requires_grad with nested list."""
        tensor1 = paddle.randn([2, 3])
        tensor1.stop_gradient = False
        tensor2 = paddle.randn([2, 3])
        tensor2.stop_gradient = True
        inputs = [[tensor1], [tensor2]]
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], list)
        self.assertFalse(result[0][0].stop_gradient)
        self.assertTrue(result[1][0].stop_gradient)

    def test_detach_list_with_none(self):
        """Test detach_and_requires_grad with list containing None."""
        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = False
        inputs = [tensor, None, "string_value", 123]
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)
        self.assertFalse(result[0].stop_gradient)
        self.assertIsNone(result[1])
        self.assertEqual(result[2], "string_value")
        self.assertEqual(result[3], 123)

    def test_detach_tuple_with_none(self):
        """Test detach_and_requires_grad with tuple containing None."""
        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = False
        inputs = (tensor, None, "string")
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        self.assertFalse(result[0].stop_gradient)
        self.assertIsNone(result[1])
        self.assertEqual(result[2], "string")

    def test_detach_dict(self):
        """Test detach_and_requires_grad with dict."""
        tensor1 = paddle.randn([2, 3])
        tensor1.stop_gradient = False
        tensor2 = paddle.randn([2, 3])
        tensor2.stop_gradient = True
        inputs = {"key1": tensor1, "key2": tensor2}
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertFalse(result["key1"].stop_gradient)
        self.assertTrue(result["key2"].stop_gradient)

    def test_detach_dict_with_none(self):
        """Test detach_and_requires_grad with dict containing None."""
        tensor = paddle.randn([2, 3])
        tensor.stop_gradient = False
        inputs = {"key1": tensor, "key2": None}
        result = detach_and_requires_grad(inputs)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertFalse(result["key1"].stop_gradient)
        self.assertIsNone(result["key2"])


class TestCloneAndClearDataptr(unittest.TestCase):
    def test_clone_single_tensor(self):
        """Test clone_and_clear_dataptr with a single tensor."""
        tensor = paddle.randn([2, 3])
        result = clone_and_clear_dataptr(tensor, clear_dataptr=False)

        self.assertIsInstance(result, paddle.Tensor)
        self.assertEqual(result.shape, tensor.shape)

    def test_clone_single_tensor_clear_dataptr(self):
        """Test clone_and_clear_dataptr with clear_dataptr=True."""
        tensor = paddle.randn([2, 3])
        result = clone_and_clear_dataptr(tensor, clear_dataptr=True)

        self.assertIsInstance(result, paddle.Tensor)
        # After _clear_dataptr(), the shape may be cleared

    def test_clone_list_of_tensors(self):
        """Test clone_and_clear_dataptr with a list of tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = [tensor1, tensor2]
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, tensor1.shape)
        self.assertEqual(result[1].shape, tensor2.shape)

    def test_clone_list_clear_dataptr(self):
        """Test clone_and_clear_dataptr with list and clear_dataptr=True."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = [tensor1, tensor2]
        result = clone_and_clear_dataptr(outputs, clear_dataptr=True)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_clone_tuple_of_tensors(self):
        """Test clone_and_clear_dataptr with a tuple of tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = (tensor1, tensor2)
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, tensor1.shape)
        self.assertEqual(result[1].shape, tensor2.shape)

    def test_clone_tuple_clear_dataptr(self):
        """Test clone_and_clear_dataptr with tuple and clear_dataptr=True."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = (tensor1, tensor2)
        result = clone_and_clear_dataptr(outputs, clear_dataptr=True)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_clone_list_with_none(self):
        """Test clone_and_clear_dataptr with list containing None and non-tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = [tensor1, None, "string", tensor2, 123]
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Only tensors are included
        self.assertEqual(result[0].shape, tensor1.shape)
        self.assertEqual(result[1].shape, tensor2.shape)

    def test_clone_tuple_with_none(self):
        """Test clone_and_clear_dataptr with tuple containing None and non-tensors."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = (tensor1, None, "string", tensor2)
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # Only tensors are included
        self.assertEqual(result[0].shape, tensor1.shape)
        self.assertEqual(result[1].shape, tensor2.shape)

    def test_clone_dict(self):
        """Test clone_and_clear_dataptr with dict."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = {"key1": tensor1, "key2": tensor2}
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn("key1", result)
        self.assertIn("key2", result)
        self.assertEqual(result["key1"].shape, tensor1.shape)
        self.assertEqual(result["key2"].shape, tensor2.shape)

    def test_clone_dict_clear_dataptr(self):
        """Test clone_and_clear_dataptr with dict and clear_dataptr=True."""
        tensor1 = paddle.randn([2, 3])
        tensor2 = paddle.randn([2, 3])
        outputs = {"key1": tensor1, "key2": tensor2}
        result = clone_and_clear_dataptr(outputs, clear_dataptr=True)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)

    def test_clone_dict_with_none(self):
        """Test clone_and_clear_dataptr with dict containing None and non-tensors."""
        tensor = paddle.randn([2, 3])
        outputs = {"key1": tensor, "key2": None, "key3": "value"}
        result = clone_and_clear_dataptr(outputs, clear_dataptr=False)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)  # Only tensors are included
        self.assertIn("key1", result)
        self.assertEqual(result["key1"].shape, tensor.shape)


if __name__ == "__main__":
    unittest.main()
