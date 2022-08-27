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

import numpy as np
import paddle

import paddle.audio
from audio_base import get_wav_data


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.initParmas()

    def initParmas(self):
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
        self.waveform = np.loadtxt('testdata/audio.txt')

    def test_stft(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram
        feature_librosa = np.load('testdata/librosa_stft.npy')
        x = paddle.to_tensor(self.waveform).unsqueeze(0)
        window = paddle.audio.functional.get_window(self.window_str,
                                                    self.n_fft,
                                                    dtype=x.dtype)
        feature_paddle = paddle.signal.stft(
            x=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=window,
            center=True,
            pad_mode=self.pad_mode,
            normalized=False,
            onesided=True,
        ).squeeze(0)

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_paddle,
                                             decimal=5)

    def test_istft(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram
        # librosa
        stft_matrix = np.load('testdata/librosa_stft_matrix.npy')
        feature_librosa = np.loadtxt('testdata/librosa_istft.txt')

        x = paddle.to_tensor(stft_matrix).unsqueeze(0)
        window = paddle.audio.functional.get_window(self.window_str,
                                                    self.n_fft,
                                                    dtype=paddle.to_tensor(
                                                        self.waveform).dtype)
        feature_paddle = paddle.signal.istft(
            x=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=None,
            window=window,
            center=True,
            normalized=False,
            onesided=True,
            length=None,
            return_complex=False,
        ).squeeze(0)

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_paddle,
                                             decimal=5)

    def test_mel(self):
        feature_librosa = np.loadtxt('testdata/librosa_filters_mel.txt')
        feature_compliance = paddle.audio.compliance.librosa.compute_fbank_matrix(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=None,
            htk=False,
            norm='slaney',
            dtype=self.waveform.dtype,
        )
        x = paddle.to_tensor(self.waveform)
        feature_functional = paddle.audio.functional.compute_fbank_matrix(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=None,
            htk=False,
            norm='slaney',
            dtype=x.dtype,
        )

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_compliance)
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_functional)

    def test_melspect(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = np.loadtxt('testdata/librosa_melspectrogram.txt')

        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.melspectrogram(
            x=self.waveform,
            sr=self.sr,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            to_db=False)

        # paddle.audio.features.layer
        x = paddle.to_tensor(self.waveform, dtype=paddle.float64).unsqueeze(
            0)  # Add batch dim.
        feature_extractor = paddle.audio.features.MelSpectrogram(
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_compliance,
                                             decimal=5)
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_layer,
                                             decimal=5)

    def test_log_melspect(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = np.loadtxt('testdata/librosa_logmelspect.txt')
        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.melspectrogram(
            x=self.waveform,
            sr=self.sr,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin)

    def test_spectrogram(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram
        feature_librosa = np.loadtxt('testdata/librosa_spectrogram.txt')
        feature_compliance = paddle.audio.compliance.librosa.spectrogram(
            x=self.waveform,
            sr=self.sr,
            window=self.window_str,
            pad_mode=self.pad_mode)
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_compliance,
                                             decimal=5)

    def test_mfcc(self):
        if len(self.waveform.shape) == 2:  # (C, T)
            self.waveform = self.waveform.squeeze(
                0)  # 1D input for librosa.feature.melspectrogram

        # librosa:
        feature_librosa = np.loadtxt('testdata/librosa_mfcc.txt')
        # paddle.audio.compliance.librosa:
        feature_compliance = paddle.audio.compliance.librosa.mfcc(
            x=self.waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            dct_type=2,
            norm='ortho',
            lifter=0,
            window_size=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            top_db=self.top_db)
        # paddlespeech.audio.features.layer
        x = paddle.to_tensor(self.waveform, dtype=paddle.float64).unsqueeze(
            0)  # Add batch dim.
        feature_extractor = paddle.audio.features.MFCC(
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            top_db=self.top_db,
            dtype=x.dtype)
        feature_layer = feature_extractor(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_compliance,
                                             decimal=4)
        np.testing.assert_array_almost_equal(feature_librosa,
                                             feature_layer,
                                             decimal=4)

    def test_kaldi_feature(self):
        waveform = np.expand_dims(self.waveform, axis=0)
        fbank = paddle.audio.compliance.kaldi.fbank(paddle.to_tensor(waveform))
        fbank_bg = np.loadtxt('testdata/kaldi_fbank.txt')
        np.testing.assert_array_almost_equal(fbank, fbank_bg, decimal=4)

        mfcc = paddle.audio.compliance.kaldi.mfcc(paddle.to_tensor(waveform))
        mfcc_bg = np.loadtxt('testdata/kaldi_mfcc.txt')
        np.testing.assert_array_almost_equal(mfcc, mfcc_bg, decimal=4)

        spectrogram = paddle.audio.compliance.kaldi.spectrogram(
            paddle.to_tensor(waveform))
        spectrogram_bg = np.loadtxt('testdata/kaldi_spectrogram.txt')
        np.testing.assert_array_almost_equal(spectrogram,
                                             spectrogram_bg,
                                             decimal=4)


if __name__ == '__main__':
    unittest.main()
