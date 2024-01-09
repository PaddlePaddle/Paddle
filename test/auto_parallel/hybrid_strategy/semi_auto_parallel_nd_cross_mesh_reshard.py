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


class TestSemiAutoParallelNdCrossMeshReshard:
    def __init__(self):
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh0 = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
        self._mesh1 = dist.ProcessMesh([[4, 5], [6, 7]], dim_names=["x", "y"])
        self._dst_rank = [4, 5, 6, 7]
        self._shape = (20, 20)
        paddle.set_device(self._backend)

    def test_pp_to_rr(self):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )

        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def test_pp_to_ss(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=0, num_or_sections=2)
        if dist.get_rank() in [4, 5]:
            expect_out = paddle.split(expect_out[0], axis=1, num_or_sections=2)
        else:
            expect_out = paddle.split(expect_out[1], axis=1, num_or_sections=2)
        expect_out_shape = [10, 10]

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(0), dist.Shard(1)]
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_rr_to_pp(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
        )
        if dist.get_rank() == 4:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        if dist.get_rank() in [5, 6, 7]:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_rr_to_ss(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=0, num_or_sections=2)
        if dist.get_rank() in [4, 5]:
            expect_out = paddle.split(expect_out[0], axis=1, num_or_sections=2)
        else:
            expect_out = paddle.split(expect_out[1], axis=1, num_or_sections=2)
        expect_out_shape = [10, 10]

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Replicate(), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(0), dist.Shard(1)]
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_ss_to_pp(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Shard(1)]
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [
                dist.Partial(dist.ReduceType.kRedSum),
                dist.Partial(dist.ReduceType.kRedSum),
            ],
        )
        if dist.get_rank() == 4:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())
        if dist.get_rank() in [5, 6, 7]:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_ss_to_rr(self):
        a = paddle.ones(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Shard(1)]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Replicate(), dist.Replicate()]
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            np.testing.assert_equal(out._local_value().numpy(), a.numpy())

    def test_ss_to_ss(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=0, num_or_sections=2)
        if dist.get_rank() in [4, 5]:
            expect_out = paddle.split(expect_out[0], axis=1, num_or_sections=2)
        else:
            expect_out = paddle.split(expect_out[1], axis=1, num_or_sections=2)
        expect_out_shape = [10, 10]

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Shard(1)]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Shard(1), dist.Shard(0)]
        )

        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_sp_to_ps(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]
        b = paddle.zeros(expect_out_shape)

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Shard(0), dist.Partial(dist.ReduceType.kRedSum)],
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Shard(1)],
        )

        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            if dist.get_rank() in [4, 5]:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    expect_out[dist.get_rank() % 2].numpy(),
                )
            else:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    b.numpy(),
                )

    def test_sp_to_rs(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Shard(0), dist.Partial(dist.ReduceType.kRedSum)],
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Replicate(), dist.Shard(1)]
        )

        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_sp_to_rp(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Shard(0), dist.Partial(dist.ReduceType.kRedSum)],
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Replicate(), dist.Partial(dist.ReduceType.kRedSum)],
        )

        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            if dist.get_rank() % 2 == 0:
                np.testing.assert_equal(out._local_value().numpy(), a.numpy())
            else:
                np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_sr_to_ps(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]
        b = paddle.zeros(expect_out_shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Shard(1)],
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            if dist.get_rank() in [4, 5]:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    expect_out[dist.get_rank() % 2].numpy(),
                )
            else:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    b.numpy(),
                )

    def test_sr_to_rs(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Replicate(), dist.Shard(1)]
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_sr_to_rp(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a, self._mesh0, [dist.Shard(0), dist.Replicate()]
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Replicate(), dist.Partial(dist.ReduceType.kRedSum)],
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            if dist.get_rank() % 2 == 0:
                np.testing.assert_equal(out._local_value().numpy(), a.numpy())
            else:
                np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def test_pr_to_ps(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]
        b = paddle.zeros(expect_out_shape)

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Replicate()],
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Shard(1)],
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            if dist.get_rank() in [4, 5]:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    expect_out[dist.get_rank() % 2].numpy(),
                )
            else:
                np.testing.assert_equal(
                    out._local_value().numpy(),
                    b.numpy(),
                )

    def test_pr_to_rs(self):
        a = paddle.ones(self._shape)
        expect_out = paddle.split(a, axis=1, num_or_sections=2)
        expect_out_shape = [20, 10]

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Replicate()],
        )
        out = dist.reshard(
            input_tensor, self._mesh1, [dist.Replicate(), dist.Shard(1)]
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            assert np.equal(out._local_shape, expect_out_shape).all()
            np.testing.assert_equal(
                out._local_value().numpy(),
                expect_out[dist.get_rank() % 2].numpy(),
            )

    def test_pr_to_rp(self):
        a = paddle.ones(self._shape)
        b = paddle.zeros(self._shape)

        input_tensor = dist.shard_tensor(
            a,
            self._mesh0,
            [dist.Partial(dist.ReduceType.kRedSum), dist.Replicate()],
        )
        out = dist.reshard(
            input_tensor,
            self._mesh1,
            [dist.Replicate(), dist.Partial(dist.ReduceType.kRedSum)],
        )
        if dist.get_rank() in self._dst_rank:
            assert np.equal(out.shape, input_tensor.shape).all()
            if dist.get_rank() % 2 == 0:
                np.testing.assert_equal(out._local_value().numpy(), a.numpy())
            else:
                np.testing.assert_equal(out._local_value().numpy(), b.numpy())

    def run_test_case(self):
        self.test_pp_to_rr()
        self.test_pp_to_ss()
        self.test_rr_to_pp()
        self.test_rr_to_ss()
        self.test_ss_to_pp()
        self.test_ss_to_rr()
        self.test_ss_to_ss()
        self.test_sp_to_ps()
        self.test_sp_to_rs()
        self.test_sp_to_rp()
        self.test_sr_to_ps()
        self.test_sr_to_rs()
        self.test_sr_to_rp()
        self.test_pr_to_ps()
        self.test_pr_to_rs()
        self.test_pr_to_rp()


if __name__ == '__main__':
    TestSemiAutoParallelNdCrossMeshReshard().run_test_case()
