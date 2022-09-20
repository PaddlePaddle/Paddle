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
import unittest

import soundfile
import numpy as np
import os
import paddle.audio


class TestAudioDatasets(unittest.TestCase):

    def setUp(self):
        self.initParmas()

    def initParmas(self):

        def get_wav_data(dtype: str, num_channels: int, num_frames: int):
            dtype_ = getattr(paddle, dtype)
            base = paddle.linspace(-1.0, 1.0, num_frames, dtype=dtype_) * 0.1
            data = base.tile([num_channels, 1])
            return data

        self.duration = 0.5
        self.num_channels = 1
        self.sr = 16000
        self.dtype = "float32"
        self.window_size = 1024
        waveform_tensor = get_wav_data(self.dtype,
                                       self.num_channels,
                                       num_frames=self.duration * self.sr)
        # shape (1, 8000)
        self.waveform = waveform_tensor.numpy()

    def test_backend(self):
        base_dir = os.getcwd()
        wave_wav_path = os.path.join(base_dir, "wave_test.wav")
        paddle.audio.backends.save(wave_wav_path,
                                   paddle.to_tensor(self.waveform),
                                   self.sr,
                                   channels_first=True)

        # test backends(wave)(wave_backend) info
        wav_info = paddle.audio.backends.info(wave_wav_path)
        self.assertTrue(wav_info.sample_rate, self.sr)
        self.assertTrue(wav_info.num_channels, self.num_channels)
        self.assertTrue(wav_info.bits_per_sample, 16)

        with open(wave_wav_path, 'rb') as file_:
            wav_info = paddle.audio.backends.info(file_)
            self.assertTrue(wav_info.sample_rate, self.sr)
            self.assertTrue(wav_info.num_channels, self.num_channels)
            self.assertTrue(wav_info.bits_per_sample, 16)

        # test backends(wave_backend) load & save
        wav_data, sr = paddle.audio.backends.load(wave_wav_path)
        np.testing.assert_array_almost_equal(wav_data, self.waveform, decimal=4)
        with soundfile.SoundFile(wave_wav_path, "r") as file_:
            dtype = "float32"
            frames = file_._prepare_read(0, None, -1)
            waveform = file_.read(frames, dtype, always_2d=True)
            waveform = waveform.T
            np.testing.assert_array_almost_equal(wav_data, waveform)

        with open(wave_wav_path, 'rb') as file_:
            wav_data, sr = paddle.audio.backends.load(file_,
                                                      normalize=False,
                                                      num_frames=10000)
        with soundfile.SoundFile(wave_wav_path, "r") as file_:
            dtype = "int16"
            frames = file_._prepare_read(0, None, -1)
            waveform = file_.read(frames, dtype, always_2d=True)
            waveform = waveform.T
            np.testing.assert_array_almost_equal(wav_data, waveform)

        current_backend = paddle.audio.backends.get_current_audio_backend()
        self.assertTrue(current_backend in ["wave_backend", "soundfile"])

        backends = paddle.audio.backends.list_available_backends()
        for backend in backends:
            self.assertTrue(backend in ["wave_backend", "soundfile"])

        # Test error
        try:
            paddle.audio.backends.set_backend("jfiji")
        except NotImplementedError:
            pass

        try:
            paddle.audio.backends.save(wave_wav_path,
                                       paddle.to_tensor(self.waveform),
                                       self.sr,
                                       bits_per_sample=24,
                                       channels_first=True)
        except ValueError:
            pass

        try:
            paddle.audio.backends.save(
                wave_wav_path,
                paddle.to_tensor(self.waveform).unsqueeze(0), self.sr)
        except AssertionError:
            pass

        fake_data = np.array([0, 1, 2, 3, 4, 6], np.float32)
        soundfile.write(wave_wav_path, fake_data, 1, subtype="DOUBLE")
        try:
            wav_info = paddle.audio.backends.info(wave_wav_path)
        except NotImplementedError:
            pass
        try:
            wav_info = paddle.audio.backends.load(wave_wav_path)
        except NotImplementedError:
            pass

        if os.path.exists(wave_wav_path):
            os.remove(wave_wav_path)


if __name__ == '__main__':
    unittest.main()
