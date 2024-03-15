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


def init_process_group(strategy=None):
    nranks = paddle.distributed.ParallelEnv().nranks
    rank = dist.ParallelEnv().local_rank
    is_master = True if rank == 0 else False
    pg_group = dist.init_parallel_env()

    return pg_group.process_group


class TestProcessGroupFp32(unittest.TestCase):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float32"
        self.shape = (2, 10, 5)

    def test_create_process_group_nccl(self):
        device_id = paddle.distributed.ParallelEnv().dev_id
        paddle.set_device('gpu:%d' % device_id)

        assert paddle.distributed.is_available()

        pg = init_process_group()
        print("rank:", pg.rank(), "size:", pg.size(), "name:", pg.name())
        print("test new group api ok")

        assert paddle.distributed.get_backend() == "NCCL"

        # test allreduce sum
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

                if pg.rank() == 0:
                    dist.all_reduce(x)
                else:
                    dist.all_reduce(y)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(x_np + y_np, x_out)
                else:
                    np.testing.assert_array_equal(x_np + y_np, y_out)

            print("test allreduce sum api ok")

        # test allreduce sum with shape = []
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

                if pg.rank() == 0:
                    dist.all_reduce(x)
                else:
                    dist.all_reduce(y)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(x_np + y_np, x_out)
                else:
                    np.testing.assert_array_equal(x_np + y_np, y_out)

            print("test allreduce sum api with shape = [] ok")

        # test allreduce max
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.MAX, sync_op=False)
                    dist.wait(x)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MAX, sync_op=False)
                    dist.wait(y)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), y_out)

            print("test allreduce max api ok")

        # test allreduce max with shape = []
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.MAX)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MAX)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.maximum(x_np, y_np), y_out)

            print("test allreduce max api with shape = [] ok")

        # test allreduce min
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.MIN)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MIN)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), y_out)

            print("test allreduce min api ok")

        # test allreduce min with shape = []
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.MIN)
                else:
                    dist.all_reduce(y, dist.ReduceOp.MIN)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), x_out)
                else:
                    np.testing.assert_array_equal(np.minimum(x_np, y_np), y_out)

            print("test allreduce min api with shape [] ok")

        # test allreduce prod
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.PROD)
                else:
                    dist.all_reduce(y, dist.ReduceOp.PROD)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), x_out
                    )
                else:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), y_out
                    )

            print("test allreduce prod api ok")

        # test allreduce prod with shape = []
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

                if pg.rank() == 0:
                    dist.all_reduce(x, dist.ReduceOp.PROD)
                else:
                    dist.all_reduce(y, dist.ReduceOp.PROD)

                (x_out, y_out) = exe.run(
                    main_program,
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[x, y],
                )

                if pg.rank() == 0:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), x_out
                    )
                else:
                    np.testing.assert_array_equal(
                        np.multiply(x_np, y_np), y_out
                    )

            print("test allreduce prod api with shape = [] ok")


class TestProcessGroupFp16(TestProcessGroupFp32):
    def setUp(self):
        paddle.seed(2022)
        random.seed(2022)
        np.random.seed(2022)
        self.config()

    def config(self):
        self.dtype = "float16"
        self.shape = (4, 20, 20)


if __name__ == "__main__":
    unittest.main()
