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

import random
import unittest

import numpy as np

import paddle
import paddle.distributed as dist

paddle.enable_static()


def init_process_group(strategy=None):
    nranks = paddle.distributed.ParallelEnv().nranks
    rank = dist.ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    dist.collective._init_parallel_env("nccl")


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)
        self.rank = dist.ParallelEnv().local_rank

    @classmethod
    def setUpClass(cls):
        device_id = paddle.distributed.ParallelEnv().dev_id
        paddle.set_device('gpu:%d' % device_id)
        assert paddle.distributed.is_available()
        init_process_group()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_allreduce_sum(self):
        # rank 0
        x_np = np.random.random(self.shape).astype(self.dtype)
        # rank 1
        y_np = np.random.random(self.shape).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name="x", shape=self.shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.shape, dtype=self.dtype
                )
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x)
                else:
                    dist.all_reduce(y)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(x_np + y_np, x_out)
                else:
                    np.testing.assert_array_equal(x_np + y_np, y_out)

    def test_allreduce_sum_with_0d_input(self):
        # rank 0
        x_np = np.random.random([]).astype(self.dtype)
        # rank 1
        y_np = np.random.random([]).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[], dtype=self.dtype)
                y = paddle.static.data(name="y", shape=[], dtype=self.dtype)
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x)
                else:
                    dist.all_reduce(y)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(x_np + y_np, x_out)
                else:
                    np.testing.assert_array_equal(x_np + y_np, y_out)

    def test_allreduce_max(self):
        # rank 0
        x_np = np.random.random(self.shape).astype(self.dtype)
        # rank 1
        y_np = np.random.random(self.shape).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name="x", shape=self.shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.shape, dtype=self.dtype
                )
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.MAX)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MAX)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), y_out)

    def test_allreduce_max_with_0d_input(self):
        # rank 0
        x_np = np.random.random([]).astype(self.dtype)
        # rank 1
        y_np = np.random.random([]).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[], dtype=self.dtype)
                y = paddle.static.data(name="y", shape=[], dtype=self.dtype)
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.MAX)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MAX)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), y_out)

    def test_allreduce_min(self):
        # rank 0
        x_np = np.random.random(self.shape).astype(self.dtype)
        # rank 1
        y_np = np.random.random(self.shape).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name="x", shape=self.shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.shape, dtype=self.dtype
                )
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.MIN)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MIN)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), y_out)

    def test_allreduce_min_with_0d_input(self):
        # rank 0
        x_np = np.random.random([]).astype(self.dtype)
        # rank 1
        y_np = np.random.random([]).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[], dtype=self.dtype)
                y = paddle.static.data(name="y", shape=[], dtype=self.dtype)
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.MIN)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MIN)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), y_out)

    def test_allreduce_prod(self):
        # rank 0
        x_np = np.random.random(self.shape).astype(self.dtype)
        # rank 1
        y_np = np.random.random(self.shape).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(
                    name="x", shape=self.shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="y", shape=self.shape, dtype=self.dtype
                )
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.PROD)
                else:
                    dist.all_reduce(y, dist.ReduceOp.PROD)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), x_out
                    )
                else:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), y_out
                    )

    def test_allreduce_prod_with_0d_input(self):
        # rank 0
        x_np = np.random.random([]).astype(self.dtype)
        # rank 1
        y_np = np.random.random([]).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[], dtype=self.dtype)
                y = paddle.static.data(name="y", shape=[], dtype=self.dtype)
                exe = paddle.static.Executor()

                if self.rank == 0:
                    dist.all_reduce(x, dist.ReduceOp.PROD)
                else:
                    dist.all_reduce(y, dist.ReduceOp.PROD)

                exe.run(startup_program)
                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if self.rank == 0:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), x_out
                    )
                else:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), y_out
                    )

    def test_broadcast(self):
        # to_tensor dose not support float16 input
        if self.dtype == "float16":
            return
        # rank 0
        x_np = np.random.random(self.shape).astype(self.dtype)
        # rank 1
        y_np = np.random.random(self.shape).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                if self.rank == 0:
                    data = paddle.to_tensor(x_np)
                else:
                    data = paddle.to_tensor(y_np)
                dist.broadcast(data, 1)
                exe = paddle.static.Executor()
                exe.run(startup_program)
                (data,) = exe.run(
                    main_program,
                    feed={},
                    fetch_list=[data],
                )
                np.testing.assert_array_equal(y_np, data)

    def test_broadcast_with_0d_input(self):
        # to_tensor dose not support float16 input
        if self.dtype == "float16":
            return
        # rank 0
        x_np = np.random.random([]).astype(self.dtype)
        # rank 1
        y_np = np.random.random([]).astype(self.dtype)
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                if self.rank == 0:
                    data = paddle.to_tensor(x_np)
                else:
                    data = paddle.to_tensor(y_np)
                dist.broadcast(data, 1)
                exe = paddle.static.Executor()
                exe.run(startup_program)
                (data,) = exe.run(
                    main_program,
                    feed={},
                    fetch_list=[data],
                )
                np.testing.assert_array_equal(y_np, data)


class TestProcessGroupFp16(TestProcessGroupFp32):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float16"
        self.shape = (4, 20, 20)
        self.rank = dist.ParallelEnv().local_rank


if __name__ == "__main__":
    unittest.main()
