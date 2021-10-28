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
from paddle.fluid.ir import apply_build_strategy
import paddle.fluid as fluid
import unittest
import numpy as np


def get_resnet50_model():
    main = paddle.static.Program()
    startup = paddle.static.Program()
    with paddle.static.program_guard(main, startup):
        image = paddle.static.data(
            name="image", shape=[None, 3, 224, 224], dtype="float32")
        label = paddle.static.data(name="label", shape=[None, 1], dtype="int64")
        model = resnet50()
        loss_fn = CrossEntropyLoss()
        pred = model(image)
        loss = loss_fn(pred, label)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-3)
        optimizer.minimize(loss)

    return main, startup, image, label, loss


def global_block_contains_op(program, op_type):
    for op in program.global_block().ops:
        if op.type == op_type:
            return True
    return False


class TestApplyPassToProgram(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def test_case(self):
        main, startup, image, label, loss = get_resnet50_model()
        fused_op = "fused_elemwise_add_activation"
        self.assertFalse(global_block_contains_op(main, fused_op))
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
        self.assertTrue(global_block_contains_op(main, fused_op))


class TestIRPassBase(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        if paddle.is_compiled_with_cuda():
            fluid.set_flags({
                'FLAGS_cudnn_deterministic': 1,
                'FLAGS_max_inplace_grad_add': 6,
            })
            self.place = paddle.CUDAPlace(0)
        else:
            self.place = paddle.CPUPlace()
        self.use_cuda = isinstance(self.place, paddle.CUDAPlace)
        self.executor = paddle.static.Executor(self.place)
        self.num_classes = 1000
        self.seed = 1

    def get_strategy(self):
        return {
            'enable_inplace': True,
            'enable_addto': True,
            'fuse_all_optimizer_ops': True,
            'fuse_elewise_add_act_ops': True,
            'fuse_relu_depthwise_conv': True,
            'fuse_bn_act_ops': True,
        }

    def check_before_applied(self, main, startup):
        self.assertFalse(global_block_contains_op(main, "share_buffer"))
        self.assertFalse(global_block_contains_op(main, "coalesce_tensor"))
        self.assertFalse(
            global_block_contains_op(main, "fused_elemwise_add_activation"))

        adam_cnt = 0
        for op in main.global_block().ops:
            if op.type == "adam":
                adam_cnt += 1
        self.assertGreater(adam_cnt, 1)

    def check_after_applied(self, main, startup):
        self.assertTrue(global_block_contains_op(main, "share_buffer"))
        # fused all optimizer pass requires this
        if paddle.is_compiled_with_cuda():
            self.assertTrue(global_block_contains_op(main, "coalesce_tensor"))
        self.assertTrue(
            global_block_contains_op(main, "fused_elemwise_add_activation"))

        share_dims_cnt = 0
        non_share_dims_cnt = 0
        for op in main.global_block().ops:
            if op.type != "share_buffer":
                continue

            share_dims = op.attr("share_dims")
            if share_dims:
                for i in range(len(share_dims)):
                    self.assertEqual(share_dims[0], share_dims[i])
                if share_dims[0] is True:
                    share_dims_cnt += 1
                else:
                    non_share_dims_cnt += 1
            else:
                non_share_dims_cnt += 1
        if self.use_cuda:
            self.assertGreaterEqual(share_dims_cnt, 1)
        else:
            self.assertEqual(share_dims_cnt, 0)
        self.assertGreaterEqual(non_share_dims_cnt, 1)

        if paddle.is_compiled_with_cuda():
            adam_cnt = 0
            for op in main.global_block().ops:
                if op.type == "adam":
                    adam_cnt += 1
            self.assertEqual(adam_cnt, 1)

    def test_main(self):
        if self.use_cuda:
            batch_num = 20
            batch_size = 4
        else:
            batch_num = 3
            batch_size = 2

        paddle.seed(self.seed)
        main1, startup1, image, label, loss1 = get_resnet50_model()
        main2, startup2, image, label, loss2 = get_resnet50_model()

        build_strategy = paddle.static.BuildStrategy()
        for k, v in self.get_strategy().items():
            setattr(build_strategy, k, v)
        self.check_before_applied(main2, startup2)
        apply_build_strategy(main2, startup2, build_strategy,
                             {"use_cuda": self.use_cuda})
        self.check_after_applied(main2, startup2)

        image_shape = [batch_size] + list(image.shape)[1:]
        label_shape = [batch_size] + list(label.shape)[1:]

        paddle.seed(self.seed)
        scope1 = paddle.static.Scope()
        with paddle.static.scope_guard(scope1):
            self.executor.run(startup1)

        paddle.seed(self.seed)
        scope2 = paddle.static.Scope()
        with paddle.static.scope_guard(scope2):
            self.executor.run(startup2)

        for idx in range(batch_num):
            feed = {
                image.name: np.random.rand(*image_shape).astype('float32'),
                label.name: np.random.randint(
                    low=0,
                    high=self.num_classes,
                    size=label_shape,
                    dtype='int64'),
            }
            with paddle.static.scope_guard(scope1):
                loss_value1 = self.executor.run(main1,
                                                feed=feed,
                                                fetch_list=[loss1])[0]
            with paddle.static.scope_guard(scope2):
                loss_value2 = self.executor.run(main2,
                                                feed=feed,
                                                fetch_list=[loss2])[0]
            self.assertEqual(loss_value1, loss_value2, "batch {}".format(idx))


if __name__ == "__main__":
    unittest.main()
