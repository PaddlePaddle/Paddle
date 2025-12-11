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

import numpy as np

import paddle


def bench_split(fn1, fn2, num_warmups: int = 50, num_tests: int = 50):
    # clear
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn1()
        fn2()

    # Flush L2
    cache.zero_()
    del cache

    # Testing
    start_events_fn1 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn1 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    start_events_fn2 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn2 = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        # Record
        start_events_fn1[i].record()
        fn1()
        end_events_fn1[i].record()
        start_events_fn2[i].record()
        fn2()
        end_events_fn2[i].record()
    paddle.device.synchronize()

    times_fn1 = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn1, end_events_fn1)
        ]
    )[1:]
    times_fn2 = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn2, end_events_fn2)
        ]
    )[1:]
    return (
        np.average(times_fn1),
        np.min(times_fn1),
        np.max(times_fn1),
        np.average(times_fn2),
        np.min(times_fn2),
        np.max(times_fn2),
    )


def bench(fn, num_warmups: int = 50, num_tests: int = 50):
    # clear
    cache = paddle.empty((int(256e6 // 4),), dtype="int32")
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Flush L2
    cache.zero_()
    del cache

    # Testing
    start_events_fn = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    end_events_fn = [
        paddle.device.Event(enable_timing=True) for _ in range(num_tests)
    ]
    for i in range(num_tests):
        start_events_fn[i].record()
        fn()
        end_events_fn[i].record()
    paddle.device.synchronize()

    times_fn = np.array(
        [
            s.elapsed_time(e) / 1e3
            for s, e in zip(start_events_fn, end_events_fn)
        ]
    )[1:]
    return (
        np.average(times_fn),
        np.min(times_fn),
        np.max(times_fn),
    )


def per_token_cast_back(x_fp8: paddle.Tensor, x_scales: paddle.Tensor):
    x_fp32 = x_fp8.to("float32").view((x_fp8.shape[0], -1, 128))
    x_scales = x_scales.view((x_fp8.shape[0], -1, 1))
    return (x_fp32 * x_scales).view(x_fp8.shape).to("bfloat16")
