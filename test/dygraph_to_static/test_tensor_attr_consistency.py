# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.jit.utils import OrderedSet

DYGRAPH_ONLY_TENSOR_ATTRS_ALLOW_LIST = OrderedSet([])
STATIC_ONLY_TENSOR_ATTRS_ALLOW_LIST = OrderedSet([])


def is_magic_method(attr):
    return attr.startswith('__') and attr.endswith('__')


def get_public_attributes(cls):
    return [
        attr
        for attr in dir(cls)
        if not attr.startswith('_') or is_magic_method(attr)
    ]


class TestTensorAttrConsistency(unittest.TestCase):
    def get_tensor_dygraph_and_static_attrs(self):
        dygraph_tensor_cls = paddle.Tensor
        static_tensor_cls = paddle.pir.Value
        dygraph_tensor_attrs = get_public_attributes(dygraph_tensor_cls)
        static_tensor_attrs = get_public_attributes(static_tensor_cls)
        return OrderedSet(dygraph_tensor_attrs), OrderedSet(static_tensor_attrs)

    def test_dygraph_tensor_attr_consistency_check(self):
        (
            dygraph_tensor_attrs,
            static_tensor_attrs,
        ) = self.get_tensor_dygraph_and_static_attrs()
        dygraph_only_attrs = dygraph_tensor_attrs - static_tensor_attrs
        not_allow_dygraph_only_attrs = (
            dygraph_only_attrs - DYGRAPH_ONLY_TENSOR_ATTRS_ALLOW_LIST
        )

        self.assertEqual(
            len(not_allow_dygraph_only_attrs),
            0,
            f"Value should have same attributes as Tensor, but found dygraph only tensor attributes: {not_allow_dygraph_only_attrs}\n."
            + "If you these attributes are not supported in static graph, please add them to DYGRAPH_ONLY_TENSOR_ATTRS_ALLOW_LIST in test_dygraph_to_static/test_tensor_attr_consistency.py.",
        )

    def test_static_tensor_attr_consistency_check(self):
        (
            dygraph_tensor_attrs,
            static_tensor_attrs,
        ) = self.get_tensor_dygraph_and_static_attrs()
        static_only_attrs = static_tensor_attrs - dygraph_tensor_attrs
        not_allow_static_only_attrs = (
            static_only_attrs - STATIC_ONLY_TENSOR_ATTRS_ALLOW_LIST
        )

        self.assertEqual(
            len(not_allow_static_only_attrs),
            0,
            f"Tensor should have same attributes as Value, but found static only tensor attributes: {not_allow_static_only_attrs}\n."
            + "If you these attributes are not supported in dygraph, please add them to STATIC_ONLY_TENSOR_ATTRS_ALLOW_LIST in test_dygraph_to_static/test_tensor_attr_consistency.py.",
        )


if __name__ == '__main__':
    unittest.main()
