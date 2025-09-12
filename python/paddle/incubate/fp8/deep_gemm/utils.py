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

# The file has been adapted from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE

import os
import sys

import paddle


def bench(
    fn, num_warmups: int = 5, num_tests: int = 10, high_precision: bool = False
):
    # Flush L2 cache with 256 MB data
    paddle.device.cuda.synchronize()
    cache = paddle.empty(int(256e6 // 4), dtype=paddle.int32)
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = paddle.randn((8192, 8192), dtype=paddle.float32)
        y = paddle.randn((8192, 8192), dtype=paddle.float32)
        x @ y

    # Testing
    start_event = paddle.device.cuda.Event(enable_timing=True)
    end_event = paddle.device.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_tests):
        fn()
    end_event.record()
    paddle.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests


def get_cuda_home():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return cuda_home

    try:
        which_cmd = "which nvcc"

        nvcc_path = os.popen(which_cmd).read().strip()
        if nvcc_path:
            return os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        pass

    return None


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def calc_diff(x, y):
    x, y = x.astype(paddle.float64), y.astype(paddle.float64)
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(tensors):
    total = 0
    for t in tensors:
        if isinstance(t, tuple):
            total += count_bytes(t)
        else:
            total += t.numel() * t.element_size()
    return total
