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
import itertools
import unittest

from parameterized import parameterized
from scipy import signal

import paddle
import paddle.audio
from paddle.base import core


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)))


class TestAudioFunctions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    @parameterize(
        [
            "hamming",
            "hann",
            "triang",
            "bohman",
            "blackman",
            "cosine",
            "tukey",
            "taylor",
            "bartlett",
            "nuttall",
        ],
        [1, 512],
    )
    def test_window(self, window_type: str, n_fft: int):
        window_scipy = signal.get_window(window_type, n_fft)
        window_paddle = paddle.audio.functional.get_window(window_type, n_fft)
        window_scipy = paddle.to_tensor(window_scipy, dtype=window_paddle.dtype)
        paddle.allclose(
            window_scipy,
            window_paddle,
            atol=0.0001,
            rtol=0.0001,
        )

    @parameterize([1, 512])
    def test_window_and_exception(self, n_fft: int):
        window_scipy_gaussain = signal.windows.gaussian(n_fft, std=7)
        window_paddle_gaussian = paddle.audio.functional.get_window(
            ('gaussian', 7), n_fft, False
        )
        window_scipy_gaussain = paddle.to_tensor(
            window_scipy_gaussain, dtype=window_paddle_gaussian.dtype
        )
        paddle.allclose(
            window_scipy_gaussain,
            window_paddle_gaussian,
            atol=0.0001,
            rtol=0.0001,
        )

        window_scipy_general_gaussain = signal.windows.general_gaussian(
            n_fft, 1, 7
        )
        window_paddle_general_gaussian = paddle.audio.functional.get_window(
            ('general_gaussian', 1, 7), n_fft, False
        )
        window_scipy_general_gaussain = paddle.to_tensor(
            window_scipy_general_gaussain,
            dtype=window_paddle_general_gaussian.dtype,
        )
        paddle.allclose(
            window_scipy_gaussain,
            window_paddle_gaussian,
            atol=0.0001,
            rtol=0.0001,
        )

        window_scipy_exp = signal.windows.exponential(n_fft)
        window_paddle_exp = paddle.audio.functional.get_window(
            ('exponential', None, 1), n_fft, False
        )
        window_scipy_exp = paddle.to_tensor(
            window_scipy_exp, dtype=window_paddle_exp.dtype
        )
        paddle.allclose(
            window_scipy_exp, window_paddle_exp, atol=0.0001, rtol=0.0001
        )

        window_scipy_kaiser = signal.windows.kaiser(n_fft, beta=14.0)
        window_paddle_kaiser = paddle.audio.functional.get_window(
            ('kaiser', 14.0), n_fft
        )
        window_scipy_kaiser = paddle.to_tensor(
            window_scipy_kaiser, dtype=window_paddle_kaiser.dtype
        )
        paddle.allclose(
            window_scipy_kaiser, window_paddle_kaiser, atol=0.0001, rtol=0.0001
        )


if __name__ == '__main__':
    unittest.main()
