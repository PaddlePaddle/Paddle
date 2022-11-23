# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import unittest


def build_and_run_program(place, batch_size, beam_size, stop_gradient=False):
    fluid.default_startup_program().random_seed = 1
    fluid.default_main_program().random_seed = 1
    np.random.seed(2)

    x = layers.assign(
        np.random.rand(batch_size, beam_size, 32).astype("float32"))
    indices = fluid.data(shape=[None, beam_size], dtype="int64", name="indices")
    step_idx = layers.fill_constant(shape=[1],
                                    dtype="int64",
                                    value=0,
                                    force_cpu=True)
    max_len = layers.fill_constant(shape=[1],
                                   dtype="int64",
                                   value=10,
                                   force_cpu=True)
    cond = layers.less_than(x=step_idx, y=max_len)
    while_op = layers.While(cond)
    scores = layers.array_write(x, step_idx)
    with while_op.block():
        bs = layers.cast(layers.shape(x)[0], "int64")
        for _ in range(20):
            bs = layers.cast(bs, 'int64')
        bs.stop_gradient = stop_gradient
        batch_pos = layers.expand(
            layers.unsqueeze(layers.range(0, bs, 1, dtype=bs.dtype), [1]),
            [1, beam_size])
        topk_coordinates = layers.stack([batch_pos, indices], axis=2)
        topk_coordinates.stop_gradient = stop_gradient
        score = layers.gather_nd(x, topk_coordinates)
        layers.increment(x=step_idx, value=1.0, in_place=True)
        layers.array_write(score, i=step_idx, array=scores)
        length_cond = layers.less_than(x=step_idx, y=max_len)
        layers.assign(length_cond, cond)

    out = layers.tensor_array_to_tensor(scores, axis=0, use_stack=True)[0]
    loss = layers.reduce_mean(out)
    opt = fluid.optimizer.Adam(0.01)
    opt.minimize(loss)
    exe = fluid.Executor(place)
    data = np.random.random_integers(low=0,
                                     high=beam_size - 1,
                                     size=(batch_size,
                                           beam_size)).astype("int64")
    loss_val, = exe.run(feed={"indices": data}, fetch_list=[loss])

    return loss_val


class TestDynRNNStopGradient(unittest.TestCase):

    def setUp(self):
        self.batch_size = 20
        self.beam_size = 64

    def run_main(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            with fluid.scope_guard(fluid.Scope()):
                value1 = build_and_run_program(place, self.batch_size,
                                               self.beam_size, False)
                value2 = build_and_run_program(place, self.batch_size,
                                               self.beam_size, True)

                np.testing.assert_array_equal(value1, value2)

    def test_check_main(self):
        places = [fluid.CPUPlace()]
        if fluid.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))

        for p in places:
            self.run_main(p)


if __name__ == '__main__':
    unittest.main()
