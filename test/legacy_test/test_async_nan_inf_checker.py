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

import multiprocessing as mp
import unittest

import numpy as np

import paddle
from paddle.base.framework import _create_async_nan_inf_checker


def process_main(value):
    x = paddle.full(fill_value=value, shape=[10])
    checker = _create_async_nan_inf_checker()
    checker.check(x)
    checker.wait()


def run_process(value):
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() <= 0
    ):
        return

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=process_main, args=(value,))
    p.start()
    p.join()
    success = p.exitcode == 0
    if np.isnan(value) or np.isinf(value):
        assert not success
    else:
        assert success


class CheckNaNInfBase(unittest.TestCase):
    def test_normal(self):
        run_process(10)

    def test_nan(self):
        run_process(np.nan)

    def test_inf(self):
        run_process(np.inf)


if __name__ == "__main__":
    unittest.main()
