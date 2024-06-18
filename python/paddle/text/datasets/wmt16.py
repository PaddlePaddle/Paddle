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


import os
import tarfile
from collections import defaultdict

import numpy as np

import paddle
from paddle.dataset.common import _check_exists_and_download
from paddle.io import Dataset

__all__ = []

DATA_URL = "http://paddlemodels.bj.bcebos.com/wmt/wmt16.tar.gz"
DATA_MD5 = "0c38be43600334966403524a40dcd81e"

TOTAL_EN_WORDS = 11250
TOTAL_DE_WORDS = 19220

START_MARK = "<s>"
END_MARK = "<e>"
UNK_MARK = "<unk>"


class WMT16(Dataset):
    """
    Implementation of `WMT16 <http://www.statmt.org/wmt16/>`_ test dataset.
    ACL2016 Multimodal Machine Translation. Please see this website for more
    details: http://www.statmt.org/wmt16/multimodal-task.html#task1

    If you use the dataset created for your task, please cite the following paper:
    Multi30K: Multilingual English-German Image Descriptions.

    .. code-block:: text

        @article{elliott-EtAl:2016:VL16,
         author    = {{Elliott}, D. and {Frank}, S. and {Sima"an}, K. and {Specia}, L.},
         title     = {Multi30K: Multilingual English-German Image Descriptions},
         booktitle = {Proceedings of the 6th Workshop on Vision and Language},
         year      = {2016},
         pages     = {70--74},
         year      = 2016
        }

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None.
        mode(str): 'train', 'test' or 'val'. Default 'train'.
        src_dict_size(int): word dictionary size for source language word. Default -1.
        trg_dict_size(int): word dictionary size for target language word. Default -1.
        lang(str): source language, 'en' or 'de'. Default 'en'.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True.

    Returns:
        Dataset: Instance of WMT16 dataset. The instance of dataset has 3 fields:
            - src_ids (np.array) - The sequence of token ids of source language.
            - trg_ids (np.array) - The sequence of token ids of target language.
            - trg_ids_next (np.array) - The next sequence of token ids of target language.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.text.datasets import WMT16

            >>> class SimpleNet(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            ...     def forward(self, src_ids, trg_ids, trg_ids_next):
            ...         return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

            >>> wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)

            >>> for i in range(10):
            ...     src_ids, trg_ids, trg_ids_next = wmt16[i]
            ...     src_ids = paddle.to_tensor(src_ids)
            ...     trg_ids = paddle.to_tensor(trg_ids)
            ...     trg_ids_next = paddle.to_tensor(trg_ids_next)
            ...
            ...     model = SimpleNet()
            ...     src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
            ...     print(src_ids.item(), trg_ids.item(), trg_ids_next.item())
            89 32 33
            79 18 19
            55 26 27
            147 36 37
            106 22 23
            135 50 51
            54 43 44
            217 30 31
            146 51 52
            55 24 25
    """

    def __init__(
        self,
        data_file=None,
        mode='train',
        src_dict_size=-1,
        trg_dict_size=-1,
        lang='en',
        download=True,
    ):
        assert mode.lower() in [
            'train',
            'test',
            'val',
        ], f"mode should be 'train', 'test' or 'val', but got {mode}"
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert (
                download
            ), "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, DATA_URL, DATA_MD5, 'wmt16', download
            )

        self.lang = lang
        assert src_dict_size > 0, "dict_size should be set as positive number"
        assert trg_dict_size > 0, "dict_size should be set as positive number"
        self.src_dict_size = min(
            src_dict_size, (TOTAL_EN_WORDS if lang == "en" else TOTAL_DE_WORDS)
        )
        self.trg_dict_size = min(
            trg_dict_size, (TOTAL_DE_WORDS if lang == "en" else TOTAL_EN_WORDS)
        )

        # load source and target word dict
        self.src_dict = self._load_dict(lang, src_dict_size)
        self.trg_dict = self._load_dict(
            "de" if lang == "en" else "en", trg_dict_size
        )

        # load data
        self.data = self._load_data()

    def _load_dict(self, lang, dict_size, reverse=False):
        dict_path = os.path.join(
            paddle.dataset.common.DATA_HOME,
            "wmt16/%s_%d.dict" % (lang, dict_size),
        )
        dict_found = False
        if os.path.exists(dict_path):
            with open(dict_path, "rb") as d:
                dict_found = len(d.readlines()) == dict_size
        if not dict_found:
            self._build_dict(dict_path, dict_size, lang)

        word_dict = {}
        with open(dict_path, "rb") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip().decode()
                else:
                    word_dict[line.strip().decode()] = idx
        return word_dict

    def _build_dict(self, dict_path, dict_size, lang):
        word_dict = defaultdict(int)
        with tarfile.open(self.data_file, mode="r") as f:
            for line in f.extractfile("wmt16/train"):
                line = line.decode()
                line_split = line.strip().split("\t")
                if len(line_split) != 2:
                    continue
                sen = line_split[0] if self.lang == "en" else line_split[1]
                for w in sen.split():
                    word_dict[w] += 1

        with open(dict_path, "wb") as fout:
            fout.write((f"{START_MARK}\n{END_MARK}\n{UNK_MARK}\n").encode())
            for idx, word in enumerate(
                sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
            ):
                if idx + 3 == dict_size:
                    break
                fout.write(word[0].encode())
                fout.write(b'\n')

    def _load_data(self):
        # the index for start mark, end mark, and unk are the same in source
        # language and target language. Here uses the source language
        # dictionary to determine their indices.
        start_id = self.src_dict[START_MARK]
        end_id = self.src_dict[END_MARK]
        unk_id = self.src_dict[UNK_MARK]

        src_col = 0 if self.lang == "en" else 1
        trg_col = 1 - src_col

        self.src_ids = []
        self.trg_ids = []
        self.trg_ids_next = []
        with tarfile.open(self.data_file, mode="r") as f:
            for line in f.extractfile(f"wmt16/{self.mode}"):
                line = line.decode()
                line_split = line.strip().split("\t")
                if len(line_split) != 2:
                    continue
                src_words = line_split[src_col].split()
                src_ids = (
                    [start_id]
                    + [self.src_dict.get(w, unk_id) for w in src_words]
                    + [end_id]
                )

                trg_words = line_split[trg_col].split()
                trg_ids = [self.trg_dict.get(w, unk_id) for w in trg_words]

                trg_ids_next = trg_ids + [end_id]
                trg_ids = [start_id] + trg_ids

                self.src_ids.append(src_ids)
                self.trg_ids.append(trg_ids)
                self.trg_ids_next.append(trg_ids_next)

    def __getitem__(self, idx):
        return (
            np.array(self.src_ids[idx]),
            np.array(self.trg_ids[idx]),
            np.array(self.trg_ids_next[idx]),
        )

    def __len__(self):
        return len(self.src_ids)

    def get_dict(self, lang, reverse=False):
        """
        return the word dictionary for the specified language.

        Args:
            lang(string): A string indicating which language is the source
                          language. Available options are: "en" for English
                          and "de" for Germany.
            reverse(bool): If reverse is set to False, the returned python
                           dictionary will use word as key and use index as value.
                           If reverse is set to True, the returned python
                           dictionary will use index as key and word as value.

        Returns:
            dict: The word dictionary for the specific language.

        Examples:

            .. code-block:: python

                >>> from paddle.text.datasets import WMT16
                >>> wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)
                >>> en_dict = wmt16.get_dict('en')

        """
        dict_size = (
            self.src_dict_size if lang == self.lang else self.trg_dict_size
        )

        dict_path = os.path.join(
            paddle.dataset.common.DATA_HOME,
            "wmt16/%s_%d.dict" % (lang, dict_size),
        )
        assert os.path.exists(dict_path), "Word dictionary does not exist. "
        "Please invoke paddle.dataset.wmt16.train/test/validation first "
        "to build the dictionary."
        return self._load_dict(lang, dict_size)
