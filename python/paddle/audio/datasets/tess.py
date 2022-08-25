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

__all__ = ['TESS']


class TESS(AudioClassificationDataset):
    """
    TESS is a set of 200 target words were spoken in the carrier phrase
    "Say the word _____' by two actresses (aged 26 and 64 years) and
    recordings were made of the set portraying each of seven emotions(anger,
    disgust, fear, happiness, pleasant surprise, sadness, and neutral).
    There are 2800 stimuli in total.

    Reference:
        Toronto emotional speech set (TESS)
        https://doi.org/10.5683/SP2/E8H2MF
    """

    archieves = [
        {
            'url':
            'https://bj.bcebos.com/paddleaudio/datasets/TESS_Toronto_emotional_speech_set.zip',
            'md5':
            '1465311b24d1de704c4c63e4ccc470c7',
        },
    ]
    label_list = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'ps',  # pleasant surprise
        'sad',
    ]
    meta_info = collections.namedtuple('META_INFO',
                                       ('speaker', 'word', 'emotion'))
    audio_path = 'TESS_Toronto_emotional_speech_set'

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
        super(TESS, self).__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs)

    def _get_meta_info(self, files) -> List[collections.namedtuple]:
        ret = []
        for file in files:
            basename_without_extend = os.path.basename(file)[:-4]
            ret.append(self.meta_info(*basename_without_extend.split('_')))
        return ret

    def _get_data(self, mode, seed, n_folds,
                  split) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)):
            download_and_decompress(self.archieves, DATA_HOME)

        wav_files = []
        for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path)):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        random.seed(seed)  # shuffle samples to split data
        random.shuffle(
            wav_files
        )  # make sure using the same seed to create train and dev dataset
        meta_info = self._get_meta_info(wav_files)

        files = []
        labels = []
        n_samples_per_fold = len(meta_info) // n_folds
        for idx, sample in enumerate(meta_info):
            _, _, emotion = sample
            target = self.label_list.index(emotion)
            fold = idx // n_samples_per_fold + 1

            if mode == 'train' and int(fold) != split:
                files.append(wav_files[idx])
                labels.append(target)

            if mode != 'train' and int(fold) == split:
                files.append(wav_files[idx])
                labels.append(target)

        return files, labels
