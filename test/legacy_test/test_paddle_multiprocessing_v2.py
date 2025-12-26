# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import gc
import os
import platform
import time
import unittest

from op_test import get_device

import paddle
import paddle.incubate.multiprocessing as mp
from paddle.incubate.multiprocessing import reductions
from paddle.optimizer.fusion_utils import FusionStorage

REPEAT = 20
HAS_SHM_FILES = os.path.isdir('/dev/shm')


def fill_tensor(queue, event):
    data = queue.get()
    with paddle.no_grad():
        data[0][:] = 5
        data[1][:] = 5

    event.set()


def send_tensor(queue, event, device, dtype):
    tensor = paddle.ones([5, 5], dtype=dtype)
    queue.put(tensor)
    queue.put(tensor)
    event.wait()


def send_parambase(queue, event, device, dtype):
    tensor = paddle.nn.Layer().create_parameter(
        [5, 5],
        dtype=dtype,
        default_initializer=paddle.nn.initializer.Constant(value=1.0),
    )
    queue.put(tensor)
    queue.put(tensor)
    event.wait()


def check_ipc_tensor(event, ipc_metas):
    ground_truth1 = paddle.to_tensor([1, 2, 3])
    ground_truth2 = paddle.to_tensor([3, 4, 5])
    shared_ipc_tensor = paddle.to_tensor(
        paddle.base.core.DenseTensor._new_from_ipc(ipc_metas)
    )
    paddle.cuda.ipc_collect()

    def tensor_equal(t1, t2):
        return (t1 == t2).all().item()

    # Step1: Check initial value of ipc tensor
    while not tensor_equal(ground_truth1, shared_ipc_tensor):
        time.sleep(0.1)
    event.set()

    # Step2: Check ipc tensor after update
    while not tensor_equal(ground_truth2, shared_ipc_tensor):
        time.sleep(0.1)
    event.set()


def check_ipc_reduce(event, rebuild, meta):
    rebuilt_tensor = paddle.to_tensor(rebuild(*meta))
    event.set()


def check_fusion_storage(event, storage):
    # helper = FusionStorageHelper(
    #     storage.accumulators_meta,
    #     storage.master_weights_meta,
    #     storage.merged_model_params_meta,
    #     storage.buffer_ipc_meta,
    # )
    event.set()


class leak_checker:
    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        self.next_fds = self._get_next_fds(10)
        return self

    def __exit__(self, *args):
        if args[0] is None:
            self.test_case.assertFalse(self.has_shm_files())
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)

    def _get_next_fds(self, n=1):
        fds = [os.dup(0) for i in range(n)]
        for fd in fds:
            os.close(fd)
        return fds

    def has_shm_files(self, wait=True):
        if not HAS_SHM_FILES:
            return False
        result = self._has_shm_files()
        if result and wait:
            time.sleep(0.5)
            return self._has_shm_files()
        return result

    def _has_shm_files(self):
        gc.collect()
        names = ['paddle_' + str(pid) for pid in self.checked_pids]
        for filename in os.listdir('/dev/shm'):
            for name in names:
                if filename.startswith(name):
                    print("have", filename)
                    return True
        return False


class TestMultiprocessingBase(unittest.TestCase):
    def get_tensor(self, device="cpu"):
        self.device = device.lower()
        place = None
        tensor = paddle.zeros([5, 5], dtype="float32")
        return tensor

    def get_parameter(self):
        w = paddle.nn.Layer().create_parameter(
            [10, 10],
            default_initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        return w

    def _test_empty(self, dtype="float32"):
        q = mp.Queue()
        empty = paddle.to_tensor([], dtype=dtype)
        q.put(empty)
        out = q.get(timeout=1)
        self.assertEqual(str(out), str(empty))

    def _test_sharing(
        self, ctx=mp, device='cpu', dtype="float32", repeat=1, param=False
    ):
        def test_fill():
            if param:
                x = self.get_parameter()
                y = (x[:, 1]).detach()
            else:
                x = self.get_tensor()
                y = x[:, 1]

            data = [x, y]

            queue = ctx.Queue()
            event = ctx.Event()
            queue.put(data)

            process = ctx.Process(target=fill_tensor, args=(queue, event))
            process.daemon = True
            lc.check_pid(process.pid)
            process.start()

            event.wait(30)

            self.assertTrue(event.is_set())
            self.assertTrue(data[0].equal(5).all())
            self.assertTrue(data[1].equal(5).all())

            process.join(1 if device != get_device() else 10)
            self.assertFalse(process.is_alive())

        def test_receive():
            queue = ctx.Queue()
            event = ctx.Event()

            process = ctx.Process(
                target=send_parambase if param else send_tensor,
                args=(queue, event, device, dtype),
            )
            process.daemon = True
            lc.check_pid(process.pid)
            process.start()

            t1 = queue.get()
            t2 = queue.get()
            self.assertTrue(t1.equal(1).all())
            del t1, t2

            event.set()
            process.join(1 if device != get_device() else 10)
            self.assertFalse(process.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()


@unittest.skipIf(
    (
        not (
            paddle.is_compiled_with_cuda()
            and not paddle.is_compiled_with_rocm()
        )
        and not paddle.is_compiled_with_xpu()
    )
    or platform.system().lower() == "windows",
    "Require compiled with CUDA or XPU. Skip: ipc function on Windows is not supported.",
)
class TestMultiprocessingGpu(TestMultiprocessingBase):
    def func_test_pass_tensor(self):
        paddle.set_device(get_device())
        self._test_sharing(mp.get_context("spawn"), get_device())

    def test_pass_tensor(self):
        self.func_test_pass_tensor()

    def test_ipc_tensor(self):
        paddle.device.set_device(get_device())
        initial_tensor = paddle.to_tensor([1, 2, 3])
        bonus = paddle.to_tensor([2])
        ipc_metas = initial_tensor.value().get_tensor()._share_device_ipc()
        ctx = mp.get_context("spawn")
        event = ctx.Event()
        process = ctx.Process(target=check_ipc_tensor, args=(event, ipc_metas))
        process.daemon = True
        process.start()

        # Step1: Check initial value of ipc tensor
        event.wait(30)
        self.assertTrue(event.is_set())

        # Step2: Check ipc tensor after update
        event.clear()
        initial_tensor.add_(bonus)
        event.wait(30)
        self.assertTrue(event.is_set())

        process.join(10)
        self.assertFalse(process.is_alive())

    def test_ipc_reduce(self):
        tensor = paddle.arange(0, 64, dtype="float32").reshape([8, 8])
        dense = tensor.value().get_tensor()
        rebuild, meta = reductions._reduce_lodtensor(dense)
        ctx = mp.get_context("spawn")
        event = ctx.Event()
        process = ctx.Process(
            target=check_ipc_reduce, args=(event, rebuild, meta)
        )
        process.daemon = True
        process.start()
        event.wait(30)
        self.assertTrue(event.is_set())
        process.join(10)

    def test_fusion_storage(self):
        tensor_a = paddle.zeros([16], dtype="float32")
        tensor_b = paddle.zeros([16], dtype="float32")
        accumulators = {"momentum": {"param_a": tensor_a}}
        master_weights = {"param_b": tensor_b}
        storage = FusionStorage(
            accumulators=accumulators, master_weights=master_weights
        )
        self.assertIsNotNone(storage.buffer_ipc_meta)

        ctx = mp.get_context("spawn")
        event = ctx.Event()
        process = ctx.Process(
            target=check_fusion_storage, args=(event, storage)
        )
        process.daemon = True
        process.start()
        event.wait(30)
        self.assertTrue(event.is_set())
        process.join(10)


if __name__ == "__main__":
    unittest.main()
