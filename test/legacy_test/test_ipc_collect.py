# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# test_ipc_collect.py
import multiprocessing as mp
import time
import unittest

import paddle


def worker(tensor):
    _ = tensor.sum()

    del tensor

    paddle.device.ipc_collect()
    print(f"[Child {mp.current_process().pid}] called ipc_collect()")


class TestCudaCompat(unittest.TestCase):
    def test_ipc_collect(self):
        if not (
            paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_xpu()
        ):
            print("Skip: not compiled with CUDA/XPU.")
            return

        place = paddle.CUDAPlace(0)

        x = paddle.randn([1024, 1024, 10], dtype="float32").to(place)

        before_mem = paddle.device.cuda.memory_allocated(place)
        print(f"[Main] Before spawn: {before_mem / 1024**2:.2f} MB")

        p = mp.Process(target=worker, args=(x,))
        p.start()
        p.join()

        del x
        time.sleep(0.5)

        paddle.device.ipc_collect()
        after_mem = paddle.device.cuda.memory_allocated(place)
        print(f"[Main] After ipc_collect: {after_mem / 1024**2:.2f} MB")

        assert after_mem <= before_mem, "IPC collect did not release memory!"


mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    unittest.main()
