#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest

import paddle
from paddle import base


class TestAucOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        labels = np.random.randint(0, 2, (128, 1)).astype("int64")
        num_thresholds = 200
        slide_steps = 1

        stat_pos = np.zeros(
            (1 + slide_steps) * (num_thresholds + 1) + 1,
        ).astype("int64")
        stat_neg = np.zeros(
            (1 + slide_steps) * (num_thresholds + 1) + 1,
        ).astype("int64")

        self.inputs = {
            'Predict': pred,
            'Label': labels,
            "StatPos": stat_pos,
            "StatNeg": stat_neg,
        }
        self.attrs = {
            'curve': 'ROC',
            'num_thresholds': num_thresholds,
            "slide_steps": slide_steps,
        }

        python_auc = paddle.metric.Auc(
            name="auc", curve='ROC', num_thresholds=num_thresholds
        )
        python_auc.update(pred, labels)

        pos = python_auc._stat_pos.tolist() * 2
        pos.append(1)
        neg = python_auc._stat_neg.tolist() * 2
        neg.append(1)
        self.outputs = {
            'AUC': np.array(python_auc.accumulate()),
            'StatPosOut': np.array(pos),
            'StatNegOut': np.array(neg),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestGlobalAucOp(OpTest):
    def setUp(self):
        self.op_type = "auc"
        pred = np.random.random((128, 2)).astype("float32")
        labels = np.random.randint(0, 2, (128, 1)).astype("int64")
        num_thresholds = 200
        slide_steps = 0

        stat_pos = np.zeros((1, (num_thresholds + 1))).astype("int64")
        stat_neg = np.zeros((1, (num_thresholds + 1))).astype("int64")

        self.inputs = {
            'Predict': pred,
            'Label': labels,
            "StatPos": stat_pos,
            "StatNeg": stat_neg,
        }
        self.attrs = {
            'curve': 'ROC',
            'num_thresholds': num_thresholds,
            "slide_steps": slide_steps,
        }

        python_auc = paddle.metric.Auc(
            name="auc", curve='ROC', num_thresholds=num_thresholds
        )
        python_auc.update(pred, labels)

        pos = python_auc._stat_pos
        neg = python_auc._stat_neg
        self.outputs = {
            'AUC': np.array(python_auc.accumulate()),
            'StatPosOut': np.array([pos]),
            'StatNegOut': np.array([neg]),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestAucAPI(unittest.TestCase):
    def test_static(self):
        paddle.enable_static()
        data = paddle.static.data(name="input", shape=[-1, 1], dtype="float32")
        label = paddle.static.data(name="label", shape=[4], dtype="int64")
        ins_tag_weight = paddle.static.data(
            name="ins_tag_weight", shape=[4], dtype="float32"
        )
        result = paddle.static.auc(
            input=data, label=label, ins_tag_weight=ins_tag_weight
        )

        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        exe.run(paddle.static.default_startup_program())

        x = np.array([[0.0474], [0.5987], [0.7109], [0.9997]]).astype("float32")

        y = np.array([0, 0, 1, 0]).astype('int64')
        z = np.array([1, 1, 1, 1]).astype('float32')
        (output,) = exe.run(
            feed={"input": x, "label": y, "ins_tag_weight": z},
            fetch_list=[result[0]],
        )
        auc_np = np.array(0.66666667).astype("float32")
        np.testing.assert_allclose(output, auc_np, rtol=1e-05)
        assert auc_np.shape == auc_np.shape


class TestAucOpError(unittest.TestCase):
    def test_errors(self):
        with base.program_guard(base.Program(), base.Program()):

            def test_type1():
                data1 = paddle.static.data(
                    name="input1", shape=[-1, 2], dtype="int"
                )
                label1 = paddle.static.data(
                    name="label1", shape=[-1], dtype="int"
                )
                ins_tag_w1 = paddle.static.data(
                    name="label1", shape=[-1], dtype="int"
                )
                result1 = paddle.static.auc(
                    input=data1, label=label1, ins_tag_weight=ins_tag_w1
                )

            self.assertRaises(TypeError, test_type1)

            def test_type2():
                data2 = paddle.static.data(
                    name="input2", shape=[-1, 2], dtype="float32"
                )
                label2 = paddle.static.data(
                    name="label2", shape=[-1], dtype="float32"
                )
                result2 = paddle.static.auc(input=data2, label=label2)

            self.assertRaises(TypeError, test_type2)


if __name__ == '__main__':
    unittest.main()
