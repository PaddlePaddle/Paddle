import unittest
from collections import OrderedDict

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestScatterSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("index_put")

        x_shape = [64, 32, 1024]
        indices_shape = [16]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        index_tensor_dist_attr = TensorDistAttr()
        index_tensor_dist_attr.process_mesh = process_mesh
        self.index_spec = DistTensorSpec(
            indices_shape, index_tensor_dist_attr)
        self.indices_spec = []
        for _ in range(len(x_shape)):
            self.indices_spec.append(self.index_spec)

        self.value_spec = DistTensorSpec(self.index_spec)

        self.out_spec = DistTensorSpec(self.x_spec)

        self.attrs = OrderedDict([('accumulate', False)])

    def test_index_put_forward(self):

        # [0, 1, -1], [-1], [-1] -> [0, 1, -1]
        self.x_spec.set_dims_mapping([0, 1, -1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [0, -1, 1], [-1], [-1] -> [0, -1, 1]
        self.x_spec.set_dims_mapping([0, -1, 1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # [1, 0, -1], [-1], [-1] -> [1, 0, -1]
        self.x_spec.set_dims_mapping([1, 0, -1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1])

        # [-1, 0, 1], [-1], [-1] -> [-1, 0, 1]
        self.x_spec.set_dims_mapping([-1, 0, 1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # [1, -1, 0], [-1], [-1] -> [1, -1, 0]
        self.x_spec.set_dims_mapping([1, -1, 0])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0])

        # [-1, 1, 0], [-1], [-1] -> [-1, 1, 0]
        self.x_spec.set_dims_mapping([-1, 1, 0])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec, self.indices_spec, self.value_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1, 0])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, 0])

    def test_index_put_backward(self):

        # [0, 1, -1], [-1], [-1] -> [0, 1, -1]
        self.x_spec.set_dims_mapping([0, 1, -1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [0, -1, 1], [-1], [-1] -> [0, -1, 1]
        self.x_spec.set_dims_mapping([0, -1, 1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # [1, 0, -1], [-1], [-1] -> [1, 0, -1]
        self.x_spec.set_dims_mapping([1, 0, -1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1])

        # [-1, 0, 1], [-1], [-1] -> [-1, 0, 1]
        self.x_spec.set_dims_mapping([-1, 0, 1])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # [1, -1, 0], [-1], [-1] -> [1, -1, 0]
        self.x_spec.set_dims_mapping([1, -1, 0])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([1, -1, 0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0])

        # [-1, 1, 0], [-1], [-1] -> [-1, 1, 0]
        self.x_spec.set_dims_mapping([-1, 1, 0])
        for index_spec in self.indices_spec:
            index_spec.set_dims_mapping([-1])
        self.value_spec.set_dims_mapping([-1])
        self.out_spec.set_dims_mapping([-1, 1, 0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec, self.indices_spec, self.value_spec, self.out_spec, self.attrs['accumulate'])
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1, 0])
        for inferred_index_dist_attrs in inferred_input_dist_attrs[1]:
            self.assertEqual(inferred_index_dist_attrs.dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, 0])


if __name__ == "__main__":
    unittest.main()
