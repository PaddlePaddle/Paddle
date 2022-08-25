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
import csv
import os
import random
from typing import List

from paddle.io import Dataset
from tqdm import tqdm

from ..utils import DATA_HOME
from ..utils.download import download_and_decompress
from .dataset import feat_funcs

__all__ = ['OpenRIRNoise']


class OpenRIRNoise(Dataset):
    archieves = [
        {
            'url': 'http://www.openslr.org/resources/28/rirs_noises.zip',
            'md5': 'e6f48e257286e05de56413b4779d8ffb',
        },
    ]

    sample_rate = 16000
    meta_info = collections.namedtuple('META_INFO', ('id', 'duration', 'wav'))
    base_path = os.path.join(DATA_HOME, 'open_rir_noise')
    wav_path = os.path.join(base_path, 'RIRS_NOISES')
    csv_path = os.path.join(base_path, 'csv')
    subsets = ['rir', 'noise']

    def __init__(self,
                 subset: str='rir',
                 feat_type: str='raw',
                 target_dir=None,
                 random_chunk: bool=True,
                 chunk_duration: float=3.0,
                 seed: int=0,
                 **kwargs):

        assert subset in self.subsets, \
            'Dataset subset must be one in {}, but got {}'.format(self.subsets, subset)

        self.subset = subset
        self.feat_type = feat_type
        self.feat_config = kwargs
        self.random_chunk = random_chunk
        self.chunk_duration = chunk_duration

        OpenRIRNoise.csv_path = os.path.join(
            target_dir, "open_rir_noise",
            "csv") if target_dir else self.csv_path
        self._data = self._get_data()
        super(OpenRIRNoise, self).__init__()

        # Set up a seed to reproduce training or predicting result.
        # random.seed(seed)

    def _get_data(self):
        # Download audio files.
        print(f"rirs noises base path: {self.base_path}")
        if not os.path.isdir(self.base_path):
            download_and_decompress(
                self.archieves, self.base_path, decompress=True)
        else:
            print(
                f"{self.base_path} already exists, we will not download and decompress again"
            )

        # Data preparation.
        print(f"prepare the csv to {self.csv_path}")
        if not os.path.isdir(self.csv_path):
            os.makedirs(self.csv_path)
            self.prepare_data()

        data = []
        with open(os.path.join(self.csv_path, f'{self.subset}.csv'), 'r') as rf:
            for line in rf.readlines()[1:]:
                audio_id, duration, wav = line.strip().split(',')
                data.append(self.meta_info(audio_id, float(duration), wav))

        random.shuffle(data)
        return data

    def _convert_to_record(self, idx: int):
        sample = self._data[idx]

        record = {}
        # To show all fields in a namedtuple: `type(sample)._fields`
        for field in type(sample)._fields:
            record[field] = getattr(sample, field)

        waveform, sr = paddlespeech.audio.load(record['wav'])

        assert self.feat_type in feat_funcs.keys(), \
            f"Unknown feat_type: {self.feat_type}, it must be one in {list(feat_funcs.keys())}"
        feat_func = feat_funcs[self.feat_type]
        feat = feat_func(
            waveform, sr=sr, **self.feat_config) if feat_func else waveform

        record.update({'feat': feat})
        return record

    @staticmethod
    def _get_chunks(seg_dur, audio_id, audio_duration):
        num_chunks = int(audio_duration / seg_dur)  # all in milliseconds

        chunk_lst = [
            audio_id + "_" + str(i * seg_dur) + "_" + str(i * seg_dur + seg_dur)
            for i in range(num_chunks)
        ]
        return chunk_lst

    def _get_audio_info(self, wav_file: str,
                        split_chunks: bool) -> List[List[str]]:
        waveform, sr = paddlespeech.audio.load(wav_file)
        audio_id = wav_file.split("/open_rir_noise/")[-1].split(".")[0]
        audio_duration = waveform.shape[0] / sr

        ret = []
        if split_chunks and audio_duration > self.chunk_duration:  # Split into pieces of self.chunk_duration seconds.
            uniq_chunks_list = self._get_chunks(self.chunk_duration, audio_id,
                                                audio_duration)

            for idx, chunk in enumerate(uniq_chunks_list):
                s, e = chunk.split("_")[-2:]  # Timestamps of start and end
                start_sample = int(float(s) * sr)
                end_sample = int(float(e) * sr)
                new_wav_file = os.path.join(self.base_path,
                                            audio_id + f'_chunk_{idx+1:02}.wav')
                paddlespeech.audio.save(waveform[start_sample:end_sample], sr,
                                        new_wav_file)
                # id, duration, new_wav
                ret.append([chunk, self.chunk_duration, new_wav_file])
        else:  # Keep whole audio.
            ret.append([audio_id, audio_duration, wav_file])
        return ret

    def generate_csv(self,
                     wav_files: List[str],
                     output_file: str,
                     split_chunks: bool=True):
        print(f'Generating csv: {output_file}')
        header = ["id", "duration", "wav"]

        infos = list(
            tqdm(
                map(self._get_audio_info, wav_files, [split_chunks] * len(
                    wav_files)),
                total=len(wav_files)))

        csv_lines = []
        for info in infos:
            csv_lines.extend(info)

        with open(output_file, mode="w") as csv_f:
            csv_writer = csv.writer(
                csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            for line in csv_lines:
                csv_writer.writerow(line)

    def prepare_data(self):
        rir_list = os.path.join(self.wav_path, "real_rirs_isotropic_noises",
                                "rir_list")
        rir_files = []
        with open(rir_list, 'r') as f:
            for line in f.readlines():
                rir_file = line.strip().split(' ')[-1]
                rir_files.append(os.path.join(self.base_path, rir_file))

        noise_list = os.path.join(self.wav_path, "pointsource_noises",
                                  "noise_list")
        noise_files = []
        with open(noise_list, 'r') as f:
            for line in f.readlines():
                noise_file = line.strip().split(' ')[-1]
                noise_files.append(os.path.join(self.base_path, noise_file))

        self.generate_csv(rir_files, os.path.join(self.csv_path, 'rir.csv'))
        self.generate_csv(noise_files, os.path.join(self.csv_path, 'noise.csv'))

    def __getitem__(self, idx):
        return self._convert_to_record(idx)

    def __len__(self):
        return len(self._data)
