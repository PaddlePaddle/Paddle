# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
Dataset package.
"""

import mnist
import imikolov
import imdb
import cifar
import movielens
import conll05
import uci_housing
import sentiment
import wmt14
import wmt16
import mq2007
import flowers
import voc2012

__all__ = [
    'mnist',
    'imikolov',
    'imdb',
    'cifar',
    'movielens',
    'conll05',
    'sentiment',
    'uci_housing',
    'wmt14',
    'wmt16',
    'mq2007',
    'flowers',
    'voc2012',
]
