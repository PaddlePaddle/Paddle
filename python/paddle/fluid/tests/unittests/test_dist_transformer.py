#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
import paddle
from test_dist_base import TestDistBase


def download_files():
    url_prefix = 'http://paddle-unittest-data.bj.bcebos.com/dist_transformer/'
    vocab_url = url_prefix + 'vocab.bpe.32000'
    vocab_md5 = 'a86d345ca6e27f6591d0dccb1b9be853'
    paddle.dataset.common.download(vocab_url, 'test_dist_transformer',
                                   vocab_md5)

    local_train_url = url_prefix + 'train.tok.clean.bpe.32000.en-de'
    local_train_md5 = '033eb02b9449e6dd823f050782ac8914'
    paddle.dataset.common.download(local_train_url, 'test_dist_transformer',
                                   local_train_md5)

    train0_url = url_prefix + 'train.tok.clean.bpe.32000.en-de.train_0'
    train0_md5 = 'ddce7f602f352a0405267285379a38b1'
    paddle.dataset.common.download(train0_url, 'test_dist_transformer',
                                   train0_md5)

    train1_url = url_prefix + 'train.tok.clean.bpe.32000.en-de.train_1'
    train1_md5 = '8757798200180285b1a619cd7f408747'
    paddle.dataset.common.download(train1_url, 'test_dist_transformer',
                                   train1_md5)

    test_url = url_prefix + 'newstest2013.tok.bpe.32000.en-de'
    test_md5 = '9dd74a266dbdb25314183899f269b4a2'
    paddle.dataset.common.download(test_url, 'test_dist_transformer', test_md5)
    # cut test data for faster CI
    orig_path = os.path.join(paddle.dataset.common.DATA_HOME,
                             "test_dist_transformer",
                             "newstest2013.tok.bpe.32000.en-de")
    head_path = os.path.join(paddle.dataset.common.DATA_HOME,
                             "test_dist_transformer",
                             "newstest2013.tok.bpe.32000.en-de.cut")
    os.system("head -n10 %s > %s" % (orig_path, head_path))


class TestDistTransformer2x2Sync(TestDistBase):

    def _setup_config(self):
        self._sync_mode = True

    def test_dist_train(self):
        download_files()
        self.check_with_place("dist_transformer.py",
                              delta=1e-5,
                              check_error_log=False)


class TestDistTransformer2x2Async(TestDistBase):

    def _setup_config(self):
        self._sync_mode = False

    def test_dist_train(self):
        download_files()
        self.check_with_place("dist_transformer.py",
                              delta=1.0,
                              check_error_log=False)


if __name__ == "__main__":
    unittest.main()
