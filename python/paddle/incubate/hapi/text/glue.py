#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import copy
import collections
import io
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import DATA_HOME, md5file
from paddle.incubate.hapi.download import get_path_from_url


class TSVDataset(Dataset):
    """Common tab separated text dataset that reads text fields based on provided sample splitter
    and field separator.

    The returned dataset includes samples, each of which can either be a list of text fields
    if field_separator is specified, or otherwise a single string segment produced by the
    sample_splitter.

    Example::

        # assume `test.tsv` contains the following content:
        # Id\tFirstName\tLastName
        # a\tJiheng\tJiang
        # b\tLaoban\tZha
        # discard the first line and select the 0th and 2nd fields
        dataset = data.TSVDataset('test.tsv', num_discard_samples=1, field_indices=[0, 2])
        assert dataset[0] == ['a', 'Jiang']
        assert dataset[1] == ['b', 'Zha']

    Parameters
    ----------
    filename : str or list of str
        Path to the input text file or list of paths to the input text files.
    encoding : str, default 'utf8'
        File encoding format.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    field_separator : function or None, default Splitter('\t')
        A function that splits each sample string into list of text fields.
        If None, raw samples are returned according to `sample_splitter`.
    num_discard_samples : int, default 0
        Number of samples discarded at the head of the first file.
    field_indices : list of int or None, default None
        If set, for each sample, only fields with provided indices are selected as the output.
        Otherwise all fields are returned.
    allow_missing : bool, default False
        If set to True, no exception will be thrown if the number of fields is smaller than the
        maximum field index provided.
    """

    def __init__(self,
                 filename,
                 encoding='utf-8',
                 sample_splitter=lambda x: x.splitlines(),
                 field_separator=lambda x: x.split('\t'),
                 num_discard_samples=0,
                 field_indices=None,
                 allow_missing=False):
        assert sample_splitter, 'sample_splitter must be specified.'

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._sample_splitter = sample_splitter
        self._field_separator = field_separator
        self._num_discard_samples = num_discard_samples
        self._field_indices = field_indices
        self._allow_missing = allow_missing
        self.data = self._read()

    def _should_discard(self):
        discard = self._num_discard_samples > 0
        self._num_discard_samples -= 1
        return discard

    def _field_selector(self, fields):
        if not self._field_indices:
            return fields
        try:
            result = [fields[i] for i in self._field_indices]
        except IndexError as e:
            raise (IndexError('%s. Fields = %s' % (str(e), str(fields))))
        return result

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            samples = (s for s in self._sample_splitter(content)
                       if not self._should_discard())
            if self._field_separator:
                if not self._allow_missing:
                    samples = [
                        self._field_selector(self._field_separator(s))
                        for s in samples
                    ]
                else:
                    selected_samples = []
                    num_missing = 0
                    for s in samples:
                        try:
                            fields = self._field_separator(s)
                            selected_samples.append(
                                self._field_selector(fields))
                        except IndexError:
                            num_missing += 1
                    if num_missing > 0:
                        warnings.warn('%d incomplete samples in %s' %
                                      (num_missing, filename))
                    samples = selected_samples
            all_samples += samples
        return all_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class _GlueDataset(TSVDataset):
    URL = None
    MD5 = None
    SEGMENT_INFO = collections.namedtuple(
        'SEGMENT_INFO', ('file', 'md5', 'field_indices', 'num_discard_samples'))
    SEGMENTS = {}  # mode: file, md5, field_indices, num_discard_samples

    def __init__(self,
                 segment='train',
                 root=None,
                 return_all_fields=False,
                 **kwargs):
        if return_all_fields:
            # self.SEGMENTS = copy.deepcopy(self.__class__.SEGMENTS)
            # self.SEGMENTS[segment].field_indices = segments
            segments = copy.deepcopy(self.__class__.SEGMENTS)
            segment_info = list(segments[segment])
            segment_info[2] = None
            segments[segment] = self.SEGMENT_INFO(*segment_info)
            self.SEGMENTS = segments

        self._get_data(root, segment, **kwargs)

    def _get_data(self, root, segment, **kwargs):
        default_root = os.path.join(DATA_HOME, 'glue')
        filename, data_hash, field_indices, num_discard_samples = self.SEGMENTS[
            segment]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            path = get_path_from_url(self.URL, default_root, self.MD5)
            fullname = os.path.join(default_root, filename)
        super(_GlueDataset, self).__init__(
            fullname,
            field_indices=field_indices,
            num_discard_samples=num_discard_samples,
            **kwargs)


class GlueCoLA(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4'
    MD5 = 'b178a7c2f397b0433c39c7caf50a3543'
    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'train.tsv'),
            'c79d4693b8681800338aa044bf9e797b', (3, 1), 0),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'dev.tsv'), 'c5475ccefc9e7ca0917294b8bbda783c',
            (3, 1), 0),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('CoLA', 'test.tsv'),
            'd8721b7dedda0dcca73cebb2a9f4259f', (1, ), 1)
    }

    def get_labels(self):
        return ["0", "1"]


class GlueSST2(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8'
    MD5 = '9f81648d4199384278b86e315dac217c'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'train.tsv'),
            'da409a0a939379ed32a470bc0f7fe99a', (0, 1), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'dev.tsv'),
            '268856b487b2a31a28c0a93daaff7288', (0, 1), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('SST-2', 'test.tsv'),
            '3230e4efec76488b87877a56ae49675a', (1, ), 1)
    }

    def get_labels(self):
        return ["0", "1"]


class GlueMRPC(_GlueDataset):
    DEV_ID_URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc'
    DEV_ID_MD5 = '7ab59a1b04bd7cb773f98a0717106c9b'
    TRAIN_DATA_URL = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt'
    TRAIN_DATA_MD5 = '793daf7b6224281e75fe61c1f80afe35'
    TEST_DATA_URL = 'https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt'
    TEST_DATA_MD5 = 'e437fdddb92535b820fe8852e2df8a49'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'train.tsv'),
            'dc2dac669a113866a6480a0b10cd50bf', (3, 4, 0), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'dev.tsv'), '185958e46ba556b38c6a7cc63f3a2135',
            (3, 4, 0), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('MRPC', 'test.tsv'),
            '4825dab4b4832f81455719660b608de5', (3, 4), 1)
    }

    def _get_data(self, root, segment, **kwargs):
        default_root = os.path.join(DATA_HOME, 'glue')
        filename, data_hash, field_indices, num_discard_samples = self.SEGMENTS[
            segment]
        fullname = os.path.join(default_root,
                                filename) if root is None else os.path.join(
                                    os.path.expanduser(root), filename)
        if not os.path.exists(fullname) or (data_hash and
                                            not md5file(fullname) == data_hash):
            if root is not None:  # not specified, and no need to warn
                warnings.warn(
                    'md5 check failed for {}, download {} data to {}'.format(
                        filename, self.__class__.__name__, default_root))
            if segment in ('train', 'dev'):
                dev_id_path = get_path_from_url(
                    self.DEV_ID_URL,
                    os.path.join(default_root, 'MRPC'), self.DEV_ID_MD5)
                train_data_path = get_path_from_url(
                    self.TRAIN_DATA_URL,
                    os.path.join(default_root, 'MRPC'), self.TRAIN_DATA_MD5)
                # read dev data ids
                dev_ids = []
                with io.open(dev_id_path, encoding='utf-8') as ids_fh:
                    for row in ids_fh:
                        dev_ids.append(row.strip().split('\t'))

                # generate train and dev set
                train_path = os.path.join(default_root, 'MRPC', 'train.tsv')
                dev_path = os.path.join(default_root, 'MRPC', 'dev.tsv')
                with io.open(train_data_path, encoding='utf-8') as data_fh:
                    with io.open(train_path, 'w', encoding='utf-8') as train_fh:
                        with io.open(dev_path, 'w', encoding='utf8') as dev_fh:
                            header = data_fh.readline()
                            train_fh.write(header)
                            dev_fh.write(header)
                            for row in data_fh:
                                label, id1, id2, s1, s2 = row.strip().split(
                                    '\t')
                                example = '%s\t%s\t%s\t%s\t%s\n' % (label, id1,
                                                                    id2, s1, s2)
                                if [id1, id2] in dev_ids:
                                    dev_fh.write(example)
                                else:
                                    train_fh.write(example)
            else:
                test_data_path = get_path_from_url(
                    self.TEST_DATA_URL,
                    os.path.join(default_root, 'MRPC'), self.TEST_DATA_MD5)
                test_path = os.path.join(default_root, 'MRPC', 'test.tsv')
                with io.open(test_data_path, encoding='utf-8') as data_fh:
                    with io.open(test_path, 'w', encoding='utf-8') as test_fh:
                        header = data_fh.readline()
                        test_fh.write(
                            'index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')
                        for idx, row in enumerate(data_fh):
                            label, id1, id2, s1, s2 = row.strip().split('\t')
                            test_fh.write('%d\t%s\t%s\t%s\t%s\n' %
                                          (idx, id1, id2, s1, s2))
            root = default_root
        super(GlueMRPC, self)._get_data(root, segment, **kwargs)

    def get_labels(self):
        return ["0", "1"]


class GlueSTSB(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5'
    MD5 = 'd573676be38f1a075a5702b90ceab3de'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'train.tsv'),
            '4f7a86dde15fe4832c18e5b970998672', (7, 8, 9), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'dev.tsv'),
            '5f4d6b0d2a5f268b1b56db773ab2f1fe', (7, 8, 9), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('STS-B', 'test.tsv'),
            '339b5817e414d19d9bb5f593dd94249c', (7, 8), 1)
    }

    def get_labels(self):
        return None


class GlueQQP(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP.zip?alt=media&token=700c6acf-160d-4d89-81d1-de4191d02cb5'
    MD5 = 'f642d8eb365a5f69bd826e0d195b2660'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'train.tsv'),
            '72edcb18d89b332beb7f1d9f80f6d4c2', (3, 4, 5), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'dev.tsv'), '7e930999f2b5b5316084d17d0ca70ce9',
            (3, 4, 5), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('QQP', 'test.tsv'), '79bbb6adb26f67bc4b6e9fc480bc4044',
            (1, 2), 1)
    }

    def __init__(self, segment='train', root=None, return_all_fields=False):
        # QQP may include broken samples
        super(GlueQQP, self).__init__(
            segment, root, return_all_fields, allow_missing=True)

    def get_labels(self):
        return ["0", "1"]


class GlueMNLI(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce'
    MD5 = 'e343b4bdf53f927436d0792203b9b9ff'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'train.tsv'),
            '220192295e23b6705f3545168272c740', (8, 9, 11), 1),
        'dev_matched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'dev_matched.tsv'),
            'c3fa2817007f4cdf1a03663611a8ad23', (8, 9, 15), 1),
        'dev_mismatched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'dev_mismatched.tsv'),
            'b219e6fe74e4aa779e2f417ffe713053', (8, 9, 15), 1),
        'test_matched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'test_matched.tsv'),
            '33ea0389aedda8a43dabc9b3579684d9', (8, 9), 1),
        'test_mismatched': _GlueDataset.SEGMENT_INFO(
            os.path.join('MNLI', 'test_mismatched.tsv'),
            '7d2f60a73d54f30d8a65e474b615aeb6', (8, 9), 1),
    }

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]


class GlueQNLI(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601'
    MD5 = 'b4efd6554440de1712e9b54e14760e82'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'train.tsv'),
            '5e6063f407b08d1f7c7074d049ace94a', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'dev.tsv'), '1e81e211959605f144ba6c0ad7dc948b',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('QNLI', 'test.tsv'),
            'f2a29f83f3fe1a9c049777822b7fa8b0', (1, 2), 1)
    }

    def get_labels(self):
        return ["entailment", "not_entailment"]


class GlueRTE(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb'
    MD5 = 'bef554d0cafd4ab6743488101c638539'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'train.tsv'),
            'd2844f558d111a16503144bb37a8165f', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'dev.tsv'), '973cb4178d4534cf745a01c309d4a66c',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('RTE', 'test.tsv'), '6041008f3f3e48704f57ce1b88ad2e74',
            (1, 2), 1)
    }

    def get_labels(self):
        return ["entailment", "not_entailment"]


class GlueWNLI(_GlueDataset):
    URL = 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf'
    MD5 = 'a1b4bd2861017d302d29e42139657a42'

    SEGMENTS = {
        'train': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'train.tsv'),
            '5cdc5a87b7be0c87a6363fa6a5481fc1', (1, 2, 3), 1),
        'dev': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'dev.tsv'), 'a79a6dd5d71287bcad6824c892e517ee',
            (1, 2, 3), 1),
        'test': _GlueDataset.SEGMENT_INFO(
            os.path.join('WNLI', 'test.tsv'),
            'a18789ba4f60f6fdc8cb4237e4ba24b5', (1, 2), 1)
    }

    def get_labels(self):
        return ["0", "1"]
