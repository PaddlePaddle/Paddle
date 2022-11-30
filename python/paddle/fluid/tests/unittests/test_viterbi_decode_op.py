#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import core
import unittest
import paddle

paddle.enable_static()


class Decoder:
    def __init__(self, transitions, use_tag=True):
        self.transitions = transitions
        self.use_tag = use_tag
        self.start_idx, self.stop_idx = -1, -2

    def __call__(self, inputs, length):
        bs, seq_len, n_label = inputs.shape
        inputs_t = np.transpose(inputs, (1, 0, 2))
        trans_exp = np.expand_dims(self.transitions, axis=0)
        historys = []
        left_length = np.array(length)
        max_seq_len = np.amax(left_length)
        left_length = np.expand_dims(left_length, 1)
        alpha = (
            np.full((bs, n_label), -1e4, dtype='float32')
            if self.use_tag
            else np.zeros((bs, n_label), dtype='float32')
        )
        alpha[:, -1] = 0
        for i, logit in enumerate(inputs_t[:max_seq_len]):
            if i == 0 and not self.use_tag:
                alpha = logit
                left_length = left_length - 1
                continue
            alpha_exp = np.expand_dims(alpha, 2)
            alpha_trn_sum = alpha_exp + trans_exp
            max_res = np.amax(alpha_trn_sum, 1), np.argmax(alpha_trn_sum, 1)
            historys = historys + [max_res[1]] if i >= 1 else []
            alpha_nxt = max_res[0] + logit
            mask = left_length > 0
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            if self.use_tag:
                alpha += (left_length == 1) * trans_exp[:, self.stop_idx]
            left_length = left_length - 1
        scores, last_ids = np.amax(alpha, 1), np.argmax(alpha, 1)
        left_length = left_length[:, 0]
        last_ids_update = last_ids * (left_length >= 0)
        batch_path = [last_ids_update]
        batch_offset = np.arange(bs) * n_label
        for hist in reversed(historys):
            left_length = left_length + 1
            gather_idx = batch_offset + last_ids
            last_ids_update = np.take(hist, gather_idx) * (left_length > 0)
            mask = left_length == 0
            last_ids_update = last_ids_update * (1 - mask) + last_ids * mask
            batch_path.insert(0, last_ids_update)
            last_ids = last_ids_update + (left_length < 0) * last_ids
        batch_path = np.stack(batch_path, 1)
        return scores, batch_path


class TestViterbiOp(OpTest):
    def set_attr(self):
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"
        self.use_tag = True
        self.bz, self.len, self.ntags = 4, 8, 10

    def setUp(self):
        self.op_type = "viterbi_decode"
        self.python_api = paddle.text.viterbi_decode
        self.set_attr()
        bz, length, ntags = self.bz, self.len, self.ntags
        self.input = np.random.randn(bz, length, ntags).astype(self.dtype)
        self.trans = np.random.randn(ntags, ntags).astype(self.dtype)
        self.length = np.random.randint(1, length + 1, [bz]).astype('int64')
        decoder = Decoder(self.trans, self.use_tag)
        scores, path = decoder(self.input, self.length)
        self.inputs = {
            'Input': self.input,
            'Transition': self.trans,
            'Length': self.length,
        }
        self.attrs = {
            'include_bos_eos_tag': self.use_tag,
        }
        self.outputs = {'Scores': scores, 'Path': path}

    def test_output(self):
        self.check_output(check_eager=True)


class TestViterbiAPI(unittest.TestCase):
    def set_attr(self):
        self.use_tag = True
        self.bz, self.len, self.ntags = 4, 8, 10
        self.places = (
            [fluid.CPUPlace(), fluid.CUDAPlace(0)]
            if core.is_compiled_with_cuda()
            else [fluid.CPUPlace()]
        )

    def setUp(self):
        self.set_attr()
        bz, length, ntags = self.bz, self.len, self.ntags
        self.input = np.random.randn(bz, length, ntags).astype('float32')
        self.transitions = np.random.randn(ntags, ntags).astype('float32')
        self.length = np.random.randint(1, length + 1, [bz]).astype('int64')
        decoder = Decoder(self.transitions, self.use_tag)
        self.scores, self.path = decoder(self.input, self.length)

    def check_static_result(self, place):
        bz, length, ntags = self.bz, self.len, self.ntags
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            Input = fluid.data(
                name="Input", shape=[bz, length, ntags], dtype="float32"
            )
            Transition = fluid.data(
                name="Transition", shape=[ntags, ntags], dtype="float32"
            )
            Length = fluid.data(name="Length", shape=[bz], dtype="int64")
            decoder = paddle.text.ViterbiDecoder(Transition, self.use_tag)
            score, path = decoder(Input, Length)
            exe = fluid.Executor(place)
            feed_list = {
                "Input": self.input,
                "Transition": self.transitions,
                "Length": self.length,
            }
            fetches = exe.run(feed=feed_list, fetch_list=[score, path])
            np.testing.assert_allclose(fetches[0], self.scores, rtol=1e-5)
            np.testing.assert_allclose(fetches[1], self.path)

    def test_static_net(self):
        for place in self.places:
            self.check_static_result(place)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
