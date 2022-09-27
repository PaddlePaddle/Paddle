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
import os
import paddle
import itertools
from parameterized import parameterized


def parameterize(*params):
    return parameterized.expand(list(itertools.product(*params)))


class TestAudioDatasets(unittest.TestCase):

    @parameterize(["dev", "train"], [40, 64])
    def test_tess_dataset(self, mode: str, params: int):
        """
        TESS dataset
        Reference:
            Toronto emotional speech set (TESS) https://tspace.library.utoronto.ca/handle/1807/24487
            https://doi.org/10.5683/SP2/E8H2MF
        """
        archieve = {
            'url':
            'https://bj.bcebos.com/paddleaudio/datasets/TESS_Toronto_emotional_speech_set_lite.zip',
            'md5': '9ffb5e3adf28d4d6b787fa94bd59b975',
        }  #small part of TESS dataset for test.
        tess_dataset = paddle.audio.datasets.TESS(mode=mode,
                                                  feat_type='mfcc',
                                                  n_mfcc=params,
                                                  archieve=archieve)
        idx = np.random.randint(0, 30)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)

        tess_dataset = paddle.audio.datasets.TESS(mode=mode,
                                                  feat_type='spectrogram',
                                                  n_fft=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == (params // 2 + 1))
        self.assertTrue(0 <= elem[1] <= 6)

        tess_dataset = paddle.audio.datasets.TESS(mode="dev",
                                                  feat_type='logmelspectrogram',
                                                  n_mels=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)

        tess_dataset = paddle.audio.datasets.TESS(mode="dev",
                                                  feat_type='melspectrogram',
                                                  n_mels=params)
        elem = tess_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 6)

    @parameterize(["dev", "train"], [40, 64])
    def test_esc50_dataset(self, mode: str, params: int):
        """
        ESC50 dataset
        Reference:
            ESC: Dataset for Environmental Sound Classification
            http://dx.doi.org/10.1145/2733373.2806390
        """
        archieve = {
            'url':
            'https://bj.bcebos.com/paddleaudio/datasets/ESC-50-master-lite.zip',
            'md5': '1e9ba53265143df5b2804a743f2d1956',
        }  #small part of ESC50 dataset for test.
        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode,
                                                    feat_type='mfcc',
                                                    n_mfcc=params,
                                                    archieve=archieve)
        idx = np.random.randint(0, 6)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)

        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode,
                                                    feat_type='spectrogram',
                                                    n_fft=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == (params // 2 + 1))
        self.assertTrue(0 <= elem[1] <= 2)

        esc50_dataset = paddle.audio.datasets.ESC50(
            mode=mode, feat_type='logmelspectrogram', n_mels=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)

        esc50_dataset = paddle.audio.datasets.ESC50(mode=mode,
                                                    feat_type='melspectrogram',
                                                    n_mels=params)
        elem = esc50_dataset[idx]
        self.assertTrue(elem[0].shape[0] == params)
        self.assertTrue(0 <= elem[1] <= 2)


if __name__ == '__main__':
    unittest.main()
