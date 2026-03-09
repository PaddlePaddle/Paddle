# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


from __future__ import annotations

import itertools
import unittest

from parameterized import parameterized

import paddle


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)))


class TestAudioFunctions(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.initParams()

    def initParams(self):
        def get_wav_data(dtype: str, num_channels: int, num_frames: int):
            dtype_ = getattr(paddle, dtype)
            base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_) * 0.1
            data = base.tile([num_channels, 1])
            return data

        self.n_fft = 512
        self.hop_length = 128
        self.n_mels = 40
        self.n_mfcc = 20
        self.fmin = 0.0
        self.window_str = 'hann'
        self.pad_mode = 'reflect'
        self.top_db = 80.0
        self.duration = 0.5
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        self.window_size = 1024
        waveform_tensor = get_wav_data(
            self.dtype,
            self.num_channels,
            num_frames=int(self.duration * self.sr),
        )
        self.waveform = waveform_tensor.numpy()

    def convert_tensor_encoding(
        self,
        tensor: paddle.Tensor,
        dtype: paddle.dtype,
    ):
        """Convert input tensor with values between -1 and 1 to integer encoding
        Args:
            tensor: input tensor, assumed between -1 and 1
            dtype: desired output tensor dtype
        Returns:
            Tensor: shape of (n_channels, sample_rate * duration)
        """
        if dtype == paddle.int32:
            tensor *= (tensor > 0) * 2147483647 + (tensor < 0) * 2147483648
        if dtype == paddle.int16:
            tensor *= (tensor > 0) * 32767 + (tensor < 0) * 32768
        if dtype == paddle.uint8:
            tensor *= (tensor > 0) * 127 + (tensor < 0) * 128
            tensor += 128
        tensor = tensor.astype(dtype)
        return tensor

    def get_whitenoise(
        self,
        *,
        sample_rate: int = 16000,
        duration: float = 1,  # seconds
        n_channels: int = 1,
        seed: int = 0,
        dtype: str | paddle.dtype = "float32",
        channels_first=True,
        scale_factor: float = 1,
    ):
        """Generate pseudo audio data with whitenoise
        Args:
            sample_rate: Sampling rate
            duration: Length of the resulting Tensor in seconds.
            n_channels: Number of channels
            seed: Seed value used for random number generation.
                Note that this function does not modify global random generator state.
            dtype: paddle dtype
            device: device
            channels_first: whether first dimension is n_channels
            scale_factor: scale the Tensor before clamping and quantization
        Returns:
            Tensor: shape of (n_channels, sample_rate * duration)
        """
        if isinstance(dtype, str):
            dtype = getattr(paddle, dtype)
        if dtype not in [
            paddle.float64,
            paddle.float32,
            paddle.int32,
            paddle.int16,
            paddle.uint8,
        ]:
            raise NotImplementedError(f"dtype {dtype} is not supported.")

        paddle.seed(seed)
        tensor = paddle.randn(
            [n_channels, int(sample_rate * duration)], dtype=paddle.float32
        )
        tensor /= 2.0
        tensor *= scale_factor
        tensor.clip_(-1.0, 1.0)
        if not channels_first:
            tensor = tensor.T

        return self.convert_tensor_encoding(tensor, dtype)

    @parameterize(
        ["sinc_interp_hann", "sinc_interp_kaiser"],
        [16000, 44100],
    )
    def test_resample_identity(self, resampling_method, sample_rate):
        """When sampling rate is not changed, the transform returns an identical Tensor"""
        waveform = self.get_whitenoise(sample_rate=sample_rate, duration=1)

        resampled = paddle.audio.functional.resample(
            waveform,
            sample_rate,
            sample_rate,
            resampling_method=resampling_method,
        )
        assert paddle.allclose(waveform, resampled)

    @parameterize([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_downsample_size(self, resampling_method):
        sr = 16000
        waveform = self.get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        downsampled = paddle.audio.functional.resample(
            waveform, sr, sr // 2, resampling_method=resampling_method
        )
        assert downsampled.shape[-1] == waveform.shape[-1] // 2

    @parameterize([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_upsample_size(self, resampling_method):
        sr = 16000
        waveform = self.get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        downsampled = paddle.audio.functional.resample(
            waveform, sr, sr * 2, resampling_method=resampling_method
        )
        assert downsampled.shape[-1] == waveform.shape[-1] * 2

    @parameterize([("sinc_interp_hann"), ("sinc_interp_kaiser")])
    def test_resample_waveform_identity_shape(self, resampling_method):
        sr = 16000
        waveform = self.get_whitenoise(
            sample_rate=sr,
            duration=0.5,
        )
        resampled = paddle.audio.functional.resample(
            waveform, sr, sr, resampling_method=resampling_method
        )
        assert resampled.shape[-1] == waveform.shape[-1]

    @parameterize([0, -8000, 114.514])
    def test_resample_exceptions_sr_no_positive(self, sample_rate):

        waveform = self.get_whitenoise(
            sample_rate=16000, duration=0.5
        )  # shape: [1, 8000]

        with self.assertRaises(ValueError) as context:
            paddle.audio.functional.resample(waveform, 16000, sample_rate)
        self.assertIn(
            "integer",
            str(context.exception),
        )

    @parameterize([8000])
    def test_resample_exceptions_data_no_float(self, sample_rate):

        waveform = self.get_whitenoise(
            sample_rate=sample_rate, duration=0.5
        )  # shape: [1, 8000]

        waveform_int = waveform.astype(paddle.int16)
        with self.assertRaises(TypeError) as context:
            paddle.audio.functional.resample(
                waveform_int, sample_rate, sample_rate // 2
            )
        self.assertIn("floating point", str(context.exception))

    @parameterize(["invalid_method", "invalid_method2"])
    def test_resample_exceptions_invalid_method(self, resampling_method):

        waveform = self.get_whitenoise(
            sample_rate=16000, duration=0.5
        )  # shape: [1, 8000]

        with self.assertRaises(ValueError) as context:
            paddle.audio.functional.resample(
                waveform, 16000, 8000, resampling_method=resampling_method
            )
        self.assertIn("Invalid resampling method", str(context.exception))

    @parameterize([0, -5])
    def test_resample_exceptions_filter_width_not_positive(
        self, lowpass_filter_width
    ):
        waveform = self.get_whitenoise(
            sample_rate=16000, duration=0.5
        )  # shape: [1, 8000]
        with self.assertRaises(ValueError) as context:
            paddle.audio.functional.resample(
                waveform, 16000, 8000, lowpass_filter_width=lowpass_filter_width
            )
        self.assertIn(
            "Low pass filter width should be positive", str(context.exception)
        )

        with self.assertRaises(ValueError) as context:
            paddle.audio.functional.resample(
                waveform, 16000, 8000, lowpass_filter_width=lowpass_filter_width
            )
        self.assertIn(
            "Low pass filter width should be positive", str(context.exception)
        )


if __name__ == '__main__':
    unittest.main()
