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

import paddle
from paddle.vision.models import resnet50
from paddle.nn import CrossEntropyLoss
from paddle.fluid.framework import _apply_pass
import unittest


class TestApplyPassToProgram(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def global_block_contains_op(self, program, op_type):
        for op in program.global_block().ops:
            if op.type == op_type:
                return True
        return False

    def test_case(self):
        image = paddle.static.data(
            name="image", shape=[None, 3, 224, 224], dtype="float32")
        label = paddle.static.data(name="label", shape=[None, 1], dtype="int64")
        model = resnet50()
        loss_fn = CrossEntropyLoss()
        pred = model(image)
        loss = loss_fn(pred, label)
        optimizer = paddle.optimizer.SGD(learning_rate=1e-3)
        optimizer.minimize(loss)

        startup = paddle.static.default_startup_program()
        main = paddle.static.default_main_program()

        fused_op = "fused_elemwise_add_activation"
        self.assertFalse(self.global_block_contains_op(main, fused_op))
        attrs = {
            "int_attr": -3,
            "size_t_attr": 10,
            "float_attr": 3.25,
            "float32_attr": -4.5,
            "str_attr": "any string attr value",
        }
        attr_types = {
            "size_t_attr": "size_t",
            "float32_attr": "float32",
        }
        ret_attrs = _apply_pass(main, startup, "fuse_elewise_add_act_pass",
                                attrs, attr_types)
        self.assertEqual(attrs, ret_attrs)
        self.assertTrue(self.global_block_contains_op(main, fused_op))


if __name__ == "__main__":
    unittest.main()
