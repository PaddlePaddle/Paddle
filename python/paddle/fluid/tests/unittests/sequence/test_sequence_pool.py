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

import sys
import unittest

import numpy as np

import paddle

sys.path.append("../")
from op_test import OpTest, skip_check_grad_ci

paddle.enable_static()


def convert_to_offset(lod):
    offset = [[0] for i in lod]
    for i, level in enumerate(lod):
        for seq_len in level:
            offset[i].append(offset[i][-1] + seq_len)
    return offset


def compute_seqpool_sum(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i] : offset[level][i + 1], :]
            out[i] = sub_x.sum(axis=0)


def compute_seqpool_avg(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i] : offset[level][i + 1], :]
            out[i] = sub_x.mean(axis=0)


def compute_seqpool_sqrt(x, offset, out, pad_value=0.0):
    level = len(offset) - 1
    for i in range(len(offset[level]) - 1):
        if offset[level][i] == offset[level][i + 1]:
            out[i] = pad_value
        else:
            sub_x = x[offset[level][i] : offset[level][i + 1], :]
            seq_len = offset[level][i + 1] - offset[level][i]
            out[i] = sub_x.sum(axis=0) / np.sqrt(seq_len)


class TestSeqAvgPool(OpTest):
    def set_lod(self):
        return [[11]]

    def set_lod_data(self):
        x = np.random.uniform(0.1, 1, [11, 23]).astype('float32')
        return x

    def set_data(self):
        x = self.set_lod_data()
        lod = self.set_lod()
        level = len(lod) - 1
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)
        out = np.zeros((len(lod[level]), x.shape[1])).astype('float32')
        self.outputs = {'Out': out}
        return x, lod, offset, out

    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "AVERAGE"}
        compute_seqpool_avg(x, offset, out, self.attrs["pad_value"])

    def setUp(self):
        self.op_type = 'sequence_pool'
        x, lod, offset, out = self.set_data()
        self.compute(x, offset, out)
        if len(offset) > 1:
            self.outputs = {'Out': (out, [lod[0]])}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        # Remove MaxIndex after check_grad is refined.
        out = self.outputs['Out']
        if isinstance(out, tuple):
            out = out[0]
        self.outputs['MaxIndex'] = np.zeros(out.shape).astype('int32')
        self.check_grad(["X"], "Out", check_dygraph=False)


class TestSeqAvgPoolBatch1(TestSeqAvgPool):
    def set_lod(self):
        return [[11]]

    def set_lod_data(self):
        lod = self.set_lod()
        x, _ = self.get_sequence_batch_size_1_input(
            lod=lod, shape=[lod[0][0], 23]
        )
        return x


class TestSeqAvgPoolInstance0(TestSeqAvgPool):
    def set_lod(self):
        return [[0, 0, 4, 0, 3, 0, 0, 5, 0, 0]]

    def set_lod_data(self):
        lod = self.set_lod()
        x, _ = self.get_sequence_instance_size_0_input(
            lod=lod, shape=[sum(lod[0]), 10]
        )
        return x


class TestSeqAvgPoolLen0(TestSeqAvgPool):
    def set_lod(self):
        return [[0, 4, 0, 7, 0]]


class TestSeqAvgPoolLen0LoDLevel2(TestSeqAvgPool):
    def set_lod(self):
        return [[2, 0, 1, 2], [0, 4, 0, 7, 0]]


class TestSeqSumPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.1, 'pooltype': "SUM"}
        compute_seqpool_sum(x, offset, out, self.attrs["pad_value"])


class TestSeqSumPoolLen0(TestSeqSumPool):
    def set_lod(self):
        return [[0, 4, 0, 7, 0]]


class TestSeqSumPoolLen0LoDLevel2(TestSeqSumPool):
    def set_lod(self):
        return [[2, 0, 1, 2], [0, 4, 0, 7, 0]]


class TestSeqMaxPool(TestSeqAvgPool):
    def set_lod(self):
        return [[13]]

    def set_data(self):
        self.op_type = 'sequence_pool'
        x = np.random.uniform(0.1, 1, [13, 23]).astype('float32')
        lod = self.set_lod()
        level = len(lod) - 1
        offset = convert_to_offset(lod)
        for i in range(len(offset[level]) - 1):
            l = offset[level][i + 1] - offset[level][i]
            if l > 0:
                x[offset[level][i] + np.random.randint(l), :] += 2.0

        self.inputs = {'X': (x, lod)}

        out = np.zeros((len(lod[level]), 23)).astype('float32')
        self.outputs = {'Out': out}
        return x, lod, offset, out

    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.5, 'pooltype': "MAX"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"]
            else:
                sub_x = x[offset[level][i] : offset[level][i + 1], :]
                out[i] = np.amax(sub_x, axis=0)


class TestSeqMaxPoolLen0(TestSeqMaxPool):
    def set_lod(self):
        return [[0, 1, 1, 5, 6, 0]]


class TestSeqMaxPoolLen0LoDLevel2(TestSeqMaxPool):
    def set_lod(self):
        return [[2, 0, 3, 1], [0, 1, 1, 5, 6, 0]]


class TestSeqSqrtPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "SQRT"}
        compute_seqpool_sqrt(x, offset, out, self.attrs["pad_value"])


class TestSeqSqrtPoolLen0(TestSeqSqrtPool):
    def set_lod(self):
        return [[0, 7, 0, 2, 2, 0]]


class TestSeqSqrtPoolLen0LoDLevel2(TestSeqSqrtPool):
    def set_lod(self):
        return [[1, 2, 0, 3], [0, 7, 0, 2, 2, 0]]


class TestSeqLastPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "LAST"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"]
            else:
                sub_x = x[offset[level][i] : offset[level][i + 1], :]
                out[i] = sub_x[-1, :]


class TestSeqLastPoolLen0(TestSeqLastPool):
    def set_lod(self):
        return [[0, 3, 4, 0, 4, 0]]


class TestSeqLastPoolLen0LoDLevel2(TestSeqLastPool):
    def set_lod(self):
        return [[1, 0, 2, 3], [0, 3, 4, 0, 4, 0]]


class TestSeqFirstPool(TestSeqAvgPool):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.3, 'pooltype': "FIRST"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"]
            else:
                sub_x = x[offset[level][i] : offset[level][i + 1], :]
                out[i] = sub_x[0, :]


class TestSeqFirstPoolLen0(TestSeqFirstPool):
    def set_lod(self):
        return [[0, 2, 0, 3, 6, 0]]


class TestSeqFirstPoolLen0LoDLevel2(TestSeqFirstPool):
    def set_lod(self):
        return [[1, 0, 2, 3], [0, 2, 0, 3, 6, 0]]


class TestSeqAvgPool2D(TestSeqAvgPool):
    def set_lod(self):
        return [[4, 1, 3, 5]]

    def set_data(self):
        self.op_type = 'sequence_pool'
        x = np.random.uniform(0.1, 1, [13, 3, 17]).astype('float32')
        lod = self.set_lod()
        level = len(lod) - 1
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)

        out = np.zeros((len(lod[level]), 3, 17)).astype('float32')
        self.outputs = {'Out': out}
        return x, lod, offset, out

    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "AVERAGE"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 17))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 17)
                )
                out[i] = np.reshape(sub_x.mean(axis=0), (3, 17))


class TestSeqAvgPool2DLen0(TestSeqAvgPool2D):
    def set_lod(self):
        return [[0, 5, 0, 8, 0]]


class TestSeqAvgPool2DLen0LoDLevel2(TestSeqAvgPool2D):
    def set_lod(self):
        return [[1, 0, 4], [0, 5, 0, 8, 0]]


class TestSeqSumPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.2, 'pooltype': "SUM"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 17))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 17)
                )
                out[i] = np.reshape(sub_x.sum(axis=0), (3, 17))


class TestSeqSumPool2DLen0(TestSeqSumPool2D):
    def set_lod(self):
        return [[0, 8, 0, 5, 0]]


class TestSeqSumPool2DLen0LoDLevel2(TestSeqSumPool2D):
    def set_lod(self):
        return [[1, 0, 4], [0, 8, 0, 5, 0]]


class TestSeqSqrtPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "SQRT"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 17))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 17)
                )
                seq_len = offset[level][i + 1] - offset[level][i]
                out[i] = np.reshape(
                    sub_x.sum(axis=0) / np.sqrt(seq_len), (3, 17)
                )

    def test_check_grad(self):
        # Remove MaxIndex after check_grad is refined.
        out = self.outputs['Out']
        if isinstance(out, tuple):
            out = out[0]
        self.outputs['MaxIndex'] = np.zeros(out.shape).astype('int32')
        self.check_grad(
            ["X"], "Out", max_relative_error=0.06, check_dygraph=False
        )


class TestSeqSqrtPool2DLen0(TestSeqSqrtPool2D):
    def set_lod(self):
        return [[0, 8, 0, 5, 0]]


class TestSeqSqrtPool2DLen0LoDLevel2(TestSeqSqrtPool2D):
    def set_lod(self):
        return [[1, 0, 2, 2], [0, 8, 0, 5, 0]]


class TestSeqMaxPool2D(TestSeqAvgPool2D):
    def set_lod(self):
        return [[4, 1, 3, 5]]

    def set_data(self):
        self.op_type = 'sequence_pool'
        x = np.random.uniform(0.1, 1, [13, 3, 11]).astype('float32')
        lod = self.set_lod()
        level = len(lod) - 1
        self.inputs = {'X': (x, lod)}
        offset = convert_to_offset(lod)
        for i in range(len(offset[level]) - 1):
            l = offset[level][i + 1] - offset[level][i]
            if l == 0:
                continue
            x[offset[level][i] + np.random.randint(l), :] += 1.0

        out = np.zeros((len(lod[level]), 3, 11)).astype('float32')
        self.outputs = {'Out': out}
        return x, lod, offset, out

    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "MAX"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 11))
                continue
            sub_x = np.reshape(
                x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 11)
            )
            out[i] = np.reshape(np.amax(sub_x, axis=0), (3, 11))


class TestSeqMaxPool2DLen0(TestSeqMaxPool2D):
    def set_lod(self):
        return [[0, 3, 0, 10, 0]]


class TestSeqMaxPool2DLen0LoDLevel2(TestSeqMaxPool2D):
    def set_lod(self):
        return [[1, 0, 2, 2], [0, 3, 0, 10, 0]]


@skip_check_grad_ci(
    reason="Grad computation does not apply to Sequence MAX "
    "Pool executed when is_test is true."
)
class TestSeqMaxPool2DInference(TestSeqMaxPool2D):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 1.0, 'pooltype': "MAX", 'is_test': True}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 11))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 11)
                )
                out[i] = np.reshape(np.amax(sub_x, axis=0), (3, 11))

    def test_check_grad(self):
        """Grad computation does not apply to Sequence MAX
        Pool executed when is_test is true"""
        return


class TestSeqMaxPool2DInferenceLen0(TestSeqMaxPool2DInference):
    def set_lod(self):
        return [[0, 3, 0, 10, 0]]


class TestSeqMaxPool2DInferenceLen0LoDLevel2(TestSeqMaxPool2DInference):
    def set_lod(self):
        return [[1, 0, 2, 2], [0, 3, 0, 10, 0]]


class TestSeqLastPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "LAST"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 17))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 17)
                )
                out[i] = np.reshape(sub_x[-1, :], (3, 17))


class TestSeqLastPool2DLen0(TestSeqLastPool2D):
    def set_lod(self):
        return [[0, 3, 0, 1, 9, 0]]


class TestSeqLastPool2DLen0LoDLevel2(TestSeqLastPool2D):
    def set_lod(self):
        return [[1, 0, 2, 3], [0, 3, 0, 1, 9, 0]]


class TestSeqFirstPool2D(TestSeqAvgPool2D):
    def compute(self, x, offset, out):
        self.attrs = {"pad_value": 0.0, 'pooltype': "FIRST"}
        level = len(offset) - 1
        for i in range(len(offset[level]) - 1):
            if offset[level][i] == offset[level][i + 1]:
                out[i] = self.attrs["pad_value"] * np.ones((3, 17))
            else:
                sub_x = np.reshape(
                    x[offset[level][i] : offset[level][i + 1], :], (-1, 3 * 17)
                )
                out[i] = np.reshape(sub_x[0, :], (3, 17))


class TestSeqFirstPool2DLen0(TestSeqFirstPool2D):
    def set_lod(self):
        return [[0, 3, 0, 3, 7, 0]]


class TestSeqFirstPool2DLen0LoDLevel2(TestSeqFirstPool2D):
    def set_lod(self):
        return [[1, 0, 2, 3], [0, 3, 0, 3, 7, 0]]


if __name__ == '__main__':
    unittest.main()
