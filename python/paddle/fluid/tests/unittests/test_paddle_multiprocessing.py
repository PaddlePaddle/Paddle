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

import os
import gc
import unittest
import time
import paddle
import paddle.incubate.multiprocessing as mp

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

            process.join(1 if device != "gpu" else 10)
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
            process.join(1 if device != "gpu" else 10)
            self.assertFalse(process.is_alive())

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()


class TestMultiprocessingCpu(TestMultiprocessingBase):
    def func_test_pass_tensor(self):
        paddle.set_device("cpu")
        self._test_sharing(repeat=REPEAT)

    def test_pass_tensor(self):
        self.func_test_pass_tensor()

    def func_test_pass_parambase(self):
        paddle.set_device("cpu")
        self._test_sharing(repeat=1, param=True)

    def test_pass_parambase(self):
        self.func_test_pass_parambase()

    def func_test_pass_empty(self):
        paddle.set_device("cpu")
        self._test_empty()

    def test_pass_empty(self):
        self.func_test_pass_empty()


class TestMultiprocessingGpu(TestMultiprocessingBase):
    @unittest.skipIf(
        not paddle.fluid.core.is_compiled_with_cuda(),
        "core is not compiled with CUDA",
    )
    def func_test_pass_tensor(self):
        paddle.set_device("gpu")
        self._test_sharing(mp.get_context("spawn"), "gpu")

    def test_pass_tensor(self):
        self.func_test_pass_tensor()


if __name__ == "__main__":
    unittest.main()
