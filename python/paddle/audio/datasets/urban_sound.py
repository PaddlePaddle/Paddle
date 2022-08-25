# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import collections
import os
from typing import List
from typing import Tuple

from ..utils import DATA_HOME
from ..utils.download import download_and_decompress
from .dataset import AudioClassificationDataset

__all__ = ['UrbanSound8K']


class UrbanSound8K(AudioClassificationDataset):
    """
    UrbanSound8K dataset contains 8732 labeled sound excerpts (<=4s) of urban
    sounds from 10 classes: air_conditioner, car_horn, children_playing, dog_bark,
    drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The
    classes are drawn from the urban sound taxonomy.

    Reference:
        A Dataset and Taxonomy for Urban Sound Research
        https://dl.acm.org/doi/10.1145/2647868.2655045
    """

    archieves = [
        {
            'url':
            'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz',
            'md5': '9aa69802bbf37fb986f71ec1483a196e',
        },
    ]
    label_list = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark",
        "drilling", "engine_idling", "gun_shot", "jackhammer", "siren",
        "street_music"
    ]
    meta = os.path.join('UrbanSound8K', 'metadata', 'UrbanSound8K.csv')
    meta_info = collections.namedtuple(
        'META_INFO', ('filename', 'fsid', 'start', 'end', 'salience', 'fold',
                      'class_id', 'label'))
    audio_path = os.path.join('UrbanSound8K', 'audio')

    def __init__(self,
                 mode: str='train',
                 split: int=1,
                 feat_type: str='raw',
                 **kwargs):
        files, labels = self._get_data(mode, split)
        super(UrbanSound8K, self).__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs)
        """
        Ags:
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train or dev).
            split (:obj:`int`, `optional`, defaults to 1):
                It specify the fold of dev dataset.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """

    def _get_meta_info(self):
        ret = []
        with open(os.path.join(DATA_HOME, self.meta), 'r') as rf:
            for line in rf.readlines()[1:]:
                ret.append(self.meta_info(*line.strip().split(',')))
        return ret

    def _get_data(self, mode: str, split: int) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.meta)):
            download_and_decompress(self.archieves, DATA_HOME)

        meta_info = self._get_meta_info()

        files = []
        labels = []
        for sample in meta_info:
            filename, _, _, _, _, fold, target, _ = sample
            if mode == 'train' and int(fold) != split:
                files.append(
                    os.path.join(DATA_HOME, self.audio_path, f'fold{fold}',
                                 filename))
                labels.append(int(target))

            if mode != 'train' and int(fold) == split:
                files.append(
                    os.path.join(DATA_HOME, self.audio_path, f'fold{fold}',
                                 filename))
                labels.append(int(target))

        return files, labels
