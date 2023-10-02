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
import collections
import os
from typing import List, Tuple

from paddle.dataset.common import DATA_HOME
from paddle.utils import download

from .dataset import AudioClassificationDataset

__all__ = []


class TESS(AudioClassificationDataset):
    """
    TESS is a set of 200 target words were spoken in the carrier phrase
    "Say the word _____' by two actresses (aged 26 and 64 years) and
    recordings were made of the set portraying each of seven emotions(anger,
    disgust, fear, happiness, pleasant surprise, sadness, and neutral).
    There are 2800 stimuli in total.

    Reference:
        Toronto emotional speech set (TESS) https://tspace.library.utoronto.ca/handle/1807/24487
        https://doi.org/10.5683/SP2/E8H2MF

    Args:
       mode (str, optional): It identifies the dataset mode (train or dev). Defaults to train.
       n_folds (int, optional): Split the dataset into n folds. 1 fold for dev dataset and n-1 for train dataset. Defaults to 5.
       split (int, optional): It specify the fold of dev dataset. Defaults to 1.
       feat_type (str, optional): It identifies the feature type that user wants to extract of an audio file. Defaults to raw.
       archive(dict): it tells where to download the audio archive. Defaults to None.

    Returns:
        :ref:`api_paddle_io_Dataset`. An instance of TESS dataset.

    Examples:

        .. code-block:: python

            >>> import paddle

            >>> mode = 'dev'
            >>> tess_dataset = paddle.audio.datasets.TESS(mode=mode,
            ...                                         feat_type='raw')
            >>> for idx in range(5):
            ...     audio, label = tess_dataset[idx]
            ...     # do something with audio, label
            ...     print(audio.shape, label)
            ...     # [audio_data_length] , label_id

            >>> tess_dataset = paddle.audio.datasets.TESS(mode=mode,
            ...                                         feat_type='mfcc',
            ...                                         n_mfcc=40)
            >>> for idx in range(5):
            ...     audio, label = tess_dataset[idx]
            ...     # do something with mfcc feature, label
            ...     print(audio.shape, label)
            ...     # [feature_dim, num_frames] , label_id
    """

    archive = {
        'url': 'https://bj.bcebos.com/paddleaudio/datasets/TESS_Toronto_emotional_speech_set.zip',
        'md5': '1465311b24d1de704c4c63e4ccc470c7',
    }

    label_list = [
        'angry',
        'disgust',
        'fear',
        'happy',
        'neutral',
        'ps',  # pleasant surprise
        'sad',
    ]
    meta_info = collections.namedtuple(
        'META_INFO', ('speaker', 'word', 'emotion')
    )
    audio_path = 'TESS_Toronto_emotional_speech_set'

    def __init__(
        self,
        mode: str = 'train',
        n_folds: int = 5,
        split: int = 1,
        feat_type: str = 'raw',
        archive=None,
        **kwargs,
    ):
        assert isinstance(n_folds, int) and (
            n_folds >= 1
        ), f'the n_folds should be integer and n_folds >= 1, but got {n_folds}'
        assert split in range(
            1, n_folds + 1
        ), f'The selected split should be integer and should be 1 <= split <= {n_folds}, but got {split}'
        if archive is not None:
            self.archive = archive
        files, labels = self._get_data(mode, n_folds, split)
        super().__init__(
            files=files, labels=labels, feat_type=feat_type, **kwargs
        )

    def _get_meta_info(self, files) -> List[collections.namedtuple]:
        ret = []
        for file in files:
            basename_without_extend = os.path.basename(file)[:-4]
            ret.append(self.meta_info(*basename_without_extend.split('_')))
        return ret

    def _get_data(
        self, mode: str, n_folds: int, split: int
    ) -> Tuple[List[str], List[int]]:
        if not os.path.isdir(os.path.join(DATA_HOME, self.audio_path)):
            download.get_path_from_url(
                self.archive['url'],
                DATA_HOME,
                self.archive['md5'],
                decompress=True,
            )

        wav_files = []
        for root, _, files in os.walk(os.path.join(DATA_HOME, self.audio_path)):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

        meta_info = self._get_meta_info(wav_files)

        files = []
        labels = []
        for idx, sample in enumerate(meta_info):
            _, _, emotion = sample
            target = self.label_list.index(emotion)
            fold = idx % n_folds + 1

            if mode == 'train' and int(fold) != split:
                files.append(wav_files[idx])
                labels.append(target)

            if mode != 'train' and int(fold) == split:
                files.append(wav_files[idx])
                labels.append(target)

        return files, labels
