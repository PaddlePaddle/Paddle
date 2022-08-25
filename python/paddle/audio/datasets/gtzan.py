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
import random
from typing import List
from typing import Tuple

from ..utils import DATA_HOME
from ..utils.download import download_and_decompress
from .dataset import AudioClassificationDataset

__all__ = ['GTZAN']


class GTZAN(AudioClassificationDataset):
    """
    The GTZAN dataset consists of 1000 audio tracks each 30 seconds long. It contains 10 genres,
    each represented by 100 tracks. The dataset is the most-used public dataset for evaluation
    in machine listening research for music genre recognition (MGR).

    Reference:
        Musical genre classification of audio signals
        https://ieeexplore.ieee.org/document/1021072/
    """

    archieves = [
        {
            'url': 'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
            'md5': '5b3d6dddb579ab49814ab86dba69e7c7',
        },
    ]
    label_list = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal',
        'pop', 'reggae', 'rock'
    ]
    meta = os.path.join('genres', 'input.mf')
    meta_info = collections.namedtuple('META_INFO', ('file_path', 'label'))
    audio_path = 'genres'

    def __init__(self,
                 mode='train',
                 seed=0,
                 n_folds=5,
                 split=1,
                 feat_type='raw',
                 **kwargs):
        """
        Ags:
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train or dev).
            seed (:obj:`int`, `optional`, defaults to 0):
                Set the random seed to shuffle samples.
            n_folds (:obj:`int`, `optional`, defaults to 5):
                Split the dataset into n folds. 1 fold for dev dataset and n-1 for train dataset.
            split (:obj:`int`, `optional`, defaults to 1):
                It specify the fold of dev dataset.
            feat_type (:obj:`str`, `optional`, defaults to `raw`):
                It identifies the feature type that user wants to extrace of an audio file.
        """
        assert split <= n_folds, f'The selected split should not be larger than n_fold, but got {split} > {n_folds}'
        files, labels = self._get_data(mode, seed, n_folds, split)
        super(GTZAN, self).__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs)

    def _get_meta_info(self) -> List[collections.namedtuple]:
        ret = []
        with open(os.path.join(DATA_HOME, self.meta), 'r') as rf:
            for line in rf.readlines():
                ret.append(self.meta_info(*line.strip().split('\t')))
        return ret

    def _get_data(self, mode, seed, n_folds,
                  split) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)) or \
            not os.path.isfile(os.path.join(DATA_HOME, self.meta)):
            download_and_decompress(self.archieves, DATA_HOME)

        meta_info = self._get_meta_info()
        random.seed(seed)  # shuffle samples to split data
        random.shuffle(
            meta_info
        )  # make sure using the same seed to create train and dev dataset

        files = []
        labels = []
        n_samples_per_fold = len(meta_info) // n_folds
        for idx, sample in enumerate(meta_info):
            file_path, label = sample
            filename = os.path.basename(file_path)
            target = self.label_list.index(label)
            fold = idx // n_samples_per_fold + 1

            if mode == 'train' and int(fold) != split:
                files.append(
                    os.path.join(DATA_HOME, self.audio_path, label, filename))
                labels.append(target)

            if mode != 'train' and int(fold) == split:
                files.append(
                    os.path.join(DATA_HOME, self.audio_path, label, filename))
                labels.append(target)

        return files, labels
