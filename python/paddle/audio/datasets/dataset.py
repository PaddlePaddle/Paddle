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
from typing import List

import paddle

from ..features import MFCC, LogMelSpectrogram, MelSpectrogram, Spectrogram

feat_funcs = {
    'raw': None,
    'melspectrogram': MelSpectrogram,
    'mfcc': MFCC,
    'logmelspectrogram': LogMelSpectrogram,
    'spectrogram': Spectrogram,
}


class AudioClassificationDataset(paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """

    def __init__(
        self,
        files: List[str],
        labels: List[int],
        feat_type: str = 'raw',
        sample_rate: int = None,
        **kwargs,
    ):
        """
        Ags:
            files (:obj:`List[str]`): A list of absolute path of audio files.
            labels (:obj:`List[int]`): Labels of audio files.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extract an audio file.
        """
        super().__init__()

        if feat_type not in feat_funcs.keys():
            raise RuntimeError(
                f"Unknown feat_type: {feat_type}, it must be one in {list(feat_funcs.keys())}"
            )

        self.files = files
        self.labels = labels

        self.feat_type = feat_type
        self.sample_rate = sample_rate
        self.feat_config = (
            kwargs  # Pass keyword arguments to customize feature config
        )

    def _get_data(self, input_file: str):
        raise NotImplementedError

    def _convert_to_record(self, idx):
        file, label = self.files[idx], self.labels[idx]
        waveform, sample_rate = paddle.audio.load(file)
        self.sample_rate = sample_rate

        feat_func = feat_funcs[self.feat_type]

        record = {}
        if len(waveform.shape) == 2:
            waveform = waveform.squeeze(0)  # 1D input
        waveform = paddle.to_tensor(waveform, dtype=paddle.float32)
        if feat_func is not None:
            waveform = waveform.unsqueeze(0)  # (batch_size, T)
            if self.feat_type != 'spectrogram':
                feature_extractor = feat_func(
                    sr=self.sample_rate, **self.feat_config
                )
            else:
                feature_extractor = feat_func(**self.feat_config)
            record['feat'] = feature_extractor(waveform).squeeze(0)
        else:
            record['feat'] = waveform
        record['label'] = label
        return record

    def __getitem__(self, idx):
        record = self._convert_to_record(idx)
        return record['feat'], record['label']

    def __len__(self):
        return len(self.files)
