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
import json
import os
from typing import List
from typing import Tuple

from .dataset import AudioClassificationDataset

__all__ = ['HeySnips']


class HeySnips(AudioClassificationDataset):
    meta_info = collections.namedtuple('META_INFO',
                                       ('key', 'label', 'duration', 'wav'))

    def __init__(self,
                 data_dir: os.PathLike,
                 mode: str='train',
                 feat_type: str='kaldi_fbank',
                 sample_rate: int=16000,
                 **kwargs):
        self.data_dir = data_dir
        files, labels = self._get_data(mode)
        super(HeySnips, self).__init__(
            files=files,
            labels=labels,
            feat_type=feat_type,
            sample_rate=sample_rate,
            **kwargs)

    def _get_meta_info(self, mode) -> List[collections.namedtuple]:
        ret = []
        with open(os.path.join(self.data_dir, '{}.json'.format(mode)),
                  'r') as f:
            data = json.load(f)
            for item in data:
                sample = collections.OrderedDict()
                if item['duration'] > 0:
                    sample['key'] = item['id']
                    sample['label'] = 0 if item['is_hotword'] == 1 else -1
                    sample['duration'] = item['duration']
                    sample['wav'] = os.path.join(self.data_dir,
                                                 item['audio_file_path'])
                    ret.append(self.meta_info(*sample.values()))
        return ret

    def _get_data(self, mode: str) -> Tuple[List[str], List[int]]:
        meta_info = self._get_meta_info(mode)

        files = []
        labels = []
        self.keys = []
        self.durations = []
        for sample in meta_info:
            key, target, duration, wav = sample
            files.append(wav)
            labels.append(int(target))
            self.keys.append(key)
            self.durations.append(float(duration))

        return files, labels
