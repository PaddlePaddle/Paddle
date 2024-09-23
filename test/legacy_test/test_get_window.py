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
import unittest

from scipy import signal

import paddle
import paddle.audio


class TestAudioFuncitons(unittest.TestCase):
    def test_bartlett_nuttall_kaiser_window(self):
        paddle.disable_static()
        n_fft = 1

        window_scipy_bartlett = signal.windows.bartlett(n_fft)
        window_paddle_bartlett = paddle.audio.functional.get_window(
            'bartlett', n_fft
        )
        window_scipy_bartlett = paddle.to_tensor(window_scipy_bartlett, dtype=window_paddle_bartlett.dtype)
        paddle.allclose(window_scipy_bartlett, window_paddle_bartlett, atol=0.0001, rtol=0.0001)

        window_scipy_nuttall = signal.windows.nuttall(n_fft)
        window_paddle_nuttall = paddle.audio.functional.get_window(
            'nuttall', n_fft
        )
        window_scipy_nuttall = paddle.to_tensor(window_scipy_nuttall, dtype=window_paddle_nuttall.dtype)
        paddle.allclose(window_scipy_nuttall, window_paddle_nuttall, atol=0.0001, rtol=0.0001)

        window_scipy_kaiser = signal.windows.kaiser(n_fft, beta=14.0)
        window_paddle_kaiser = paddle.audio.functional.get_window(
            ('kaiser', 14.0), n_fft
        )
        window_scipy_kaiser = paddle.to_tensor(window_scipy_kaiser, dtype=window_paddle_kaiser.dtype)
        paddle.allclose(window_scipy_kaiser, window_paddle_kaiser, atol=0.0001, rtol=0.0001)

        n_fft = 512

        window_scipy_bartlett = signal.windows.bartlett(n_fft)
        window_paddle_bartlett = paddle.audio.functional.get_window(
            'bartlett', n_fft
        )
        window_scipy_bartlett = paddle.to_tensor(window_scipy_bartlett, dtype=window_paddle_bartlett.dtype)
        paddle.allclose(window_scipy_bartlett, window_paddle_bartlett, atol=0.0001, rtol=0.0001)

        window_scipy_nuttall = signal.windows.nuttall(n_fft)
        window_paddle_nuttall = paddle.audio.functional.get_window(
            'nuttall', n_fft
        )
        window_scipy_nuttall = paddle.to_tensor(window_scipy_nuttall, dtype=window_paddle_nuttall.dtype)
        paddle.allclose(window_scipy_nuttall, window_paddle_nuttall, atol=0.0001, rtol=0.0001)

        window_scipy_kaiser = signal.windows.kaiser(n_fft, beta=14.0)
        window_paddle_kaiser = paddle.audio.functional.get_window(
            ('kaiser', 14.0), n_fft
        )
        window_scipy_kaiser = paddle.to_tensor(window_scipy_kaiser, dtype=window_paddle_kaiser.dtype)
        paddle.allclose(window_scipy_kaiser, window_paddle_kaiser, atol=0.0001, rtol=0.0001)


if __name__ == '__main__':
    unittest.main()
