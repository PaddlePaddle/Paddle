# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np

import paddle
import paddle.distributed as dist


class TestSemiAutoParallelCrossMeshReshard:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh0 = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._mesh1 = dist.ProcessMesh([2, 3], dim_names=["x"])
        self._shape = (20, 20)
        self._shard_axis = 0
        self._out_shard_axis = 1
        paddle.set_device(self._backend)

    def test_p_to_r(self):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Partial(dist.ReduceType.kRedSum)]
        )
        out = dist.reshard(input_tensor, self._mesh1, [dist.Replicate()])

        if dist.get_rank() in [2, 3]:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def test_p_to_s(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(
            a, axis=self._shard_axis, num_or_sections=self._mesh1.shape[0]
        )
        expect_out_shape = list(self._shape)
        expect_out_shape[self._shard_axis] = (
            self._shape[self._shard_axis] // self._mesh1.shape[0]
        )

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Partial(dist.ReduceType.kRedSum)]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(self._shard_axis)]
        )
        if dist.get_rank() in [2, 3]:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                (
                    expect_out[0].numpy()
                    if dist.get_rank() == 2
                    else expect_out[1].numpy()
                ),
            )

    def test_r_to_p(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(a, self._mesh0, [dist.Replicate()])
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Partial(dist.ReduceType.kRedSum)]
        )
        if dist.get_rank() == 2:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        if dist.get_rank() == 3:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_r_to_s(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(
            a, axis=self._shard_axis, num_or_sections=self._mesh1.shape[0]
        )
        expect_out_shape = list(self._shape)
        expect_out_shape[self._shard_axis] = (
            self._shape[self._shard_axis] // self._mesh1.shape[0]
        )

        input_tensor = dist.shard_tensor(a, self._mesh0, [dist.Replicate()])
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(self._shard_axis)]
        )
        if dist.get_rank() in [2, 3]:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                (
                    expect_out[0].numpy()
                    if dist.get_rank() == 2
                    else expect_out[1].numpy()
                ),
            )

    def test_s_to_p(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(self._shard_axis)]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Partial(dist.ReduceType.kRedSum)]
        )
        if dist.get_rank() == 2:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        if dist.get_rank() == 3:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_s_to_r(self):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(self._shard_axis)]
        )
        out = dist.reshard(input_tensor, self._mesh1, [dist.Replicate()])
        if dist.get_rank() in [2, 3]:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def test_s_to_s(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(
            a, axis=self._out_shard_axis, num_or_sections=self._mesh1.shape[0]
        )
        expect_out_shape = list(self._shape)
        expect_out_shape[self._out_shard_axis] = (
            self._shape[self._out_shard_axis] // self._mesh1.shape[0]
        )

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(self._shard_axis)]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(self._out_shard_axis)]
        )

        if dist.get_rank() in [2, 3]:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                (
                    expect_out[0].numpy()
                    if dist.get_rank() == 2
                    else expect_out[1].numpy()
                ),
            )

    def run_test_case(self):
        self.test_p_to_r()
        self.test_p_to_s()
        self.test_r_to_p()
        self.test_r_to_s()
        self.test_s_to_p()
        self.test_s_to_r()
        self.test_s_to_s()


if __name__ == '__main__':
    TestSemiAutoParallelCrossMeshReshard().run_test_case()
