#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def sample_output_one_dimension(out, dim):
    # count numbers of different categories
    sample_prob = np.zeros(dim).astype("float32")
    sample_index_prob = np.unique(out, return_counts=True)
    sample_prob[sample_index_prob[0]] = sample_index_prob[1]
    sample_prob /= sample_prob.sum()
    return sample_prob


def sample_output_two_dimension(out, shape):
    num_dist = shape[0]
    out_list = np.split(out, num_dist, axis=0)
    sample_prob = np.zeros(shape).astype("float32")
    for i in range(num_dist):
        sample_index_prob = np.unique(out_list[i], return_counts=True)
        sample_prob[i][sample_index_prob[0]] = sample_index_prob[1]
    sample_prob /= sample_prob.sum(axis=-1, keepdims=True)
    return sample_prob


class XPUTestMultinomialOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'multinomial'
        self.use_dynamic_create_class = False

    class TestMultinomialOp(XPUOpTest):
        def setUp(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            paddle.enable_static()
            self.op_type = "multinomial"
            self.python_api = paddle.multinomial
            self.init_data()
            if self.in_type == np.uint16:
                self.inputs = {"X": convert_float_to_uint16(self.input_np)}
            else:
                self.inputs = {"X": self.input_np.astype(self.dtype)}

        def init_data(self):
            # input probability is a vector, and replacement is True
            self.input_np = np.random.rand(4).astype(np.float32)
            self.outputs = {"Out": np.zeros(100000).astype("int64")}
            self.attrs = {"num_samples": 100000, "replacement": True}

        def test_check_output(self):
            self.check_output_with_place_customized(
                self.verify_output, self.place
            )

        def sample_output(self, out):
            return sample_output_one_dimension(out, 4)

        def verify_output(self, outs):
            # normalize the input to get the probability
            prob = self.input_np / self.input_np.sum(axis=-1, keepdims=True)
            sample_prob = self.sample_output(np.array(outs[0]))
            np.testing.assert_allclose(
                sample_prob,
                prob,
                rtol=0,
                atol=0.01,
                err_msg='sample_prob: '
                + str(sample_prob)
                + '\nprob: '
                + str(prob),
            )

    class TestMultinomialOp2(TestMultinomialOp):
        def init_data(self):
            # input probability is a matrix
            self.input_np = np.random.rand(3, 4).astype(np.float32)
            self.outputs = {"Out": np.zeros((3, 100000)).astype("int64")}
            self.attrs = {"num_samples": 100000, "replacement": True}

        def sample_output(self, out):
            return sample_output_two_dimension(out, [3, 4])

    class TestMultinomialOp3(TestMultinomialOp):
        def init_data(self):
            # replacement is False. number of samples must be less than number of categories.
            self.input_np = np.random.rand(1000).astype(np.float32)
            self.outputs = {"Out": np.zeros(100).astype("int64")}
            self.attrs = {"num_samples": 100, "replacement": False}

        def verify_output(self, outs):
            out = np.array(outs[0])
            unique_out = np.unique(out)
            self.assertEqual(
                len(unique_out),
                100,
                "replacement is False. categories can't be sampled repeatedly",
            )


support_types = get_xpu_op_support_types('multinomial')
for stype in support_types:
    create_test_class(globals(), XPUTestMultinomialOp, stype)

if __name__ == "__main__":
    unittest.main()
