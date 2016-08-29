# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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

from paddle.trainer.PyDataProvider2 import *


@provider(input_types=[dense_vector(200, seq_type=SequenceType.NO_SEQUENCE)])
def test_dense_no_seq(setting, filename):
    for i in xrange(200):
        yield [(float(j - 100) * float(i + 1)) / 200.0 for j in xrange(200)]


@provider(input_types=[integer_value(200, seq_type=SequenceType.NO_SEQUENCE)])
def test_index_no_seq(setting, filename):
    for i in xrange(200):
        yield i


def test_init_hooker(setting, value, **kwargs):
    setting.value = value


@provider(input_types=[dense_vector(20, seq_type=SequenceType.NO_SEQUENCE)],
          init_hook=test_init_hooker)
def test_init_hook(setting, filename):
    for i in xrange(200):
        yield setting.value


@provider(
    input_types=[sparse_binary_vector(30000, seq_type=SequenceType.NO_SEQUENCE)])
def test_sparse_non_value_no_seq(setting, filename):
    for i in xrange(200):
        yield [(i + 1) * (j + 1) for j in xrange(10)]


@provider(input_types=[sparse_vector(30000, seq_type=SequenceType.NO_SEQUENCE)])
def test_sparse_value_no_seq(setting, filename):
    for i in xrange(200):
        yield [((i + 1) * (j + 1), float(j) / float(i + 1)) for j in xrange(10)]


@provider(input_types=[integer_value(200, seq_type=SequenceType.SEQUENCE)])
def test_index_seq(setting, filename):
    for i in xrange(200):
        yield range(i + 1)


@provider(input_types=[index_slot(200, seq_type=SequenceType.SUB_SEQUENCE)])
def test_index_sub_seq(setting, filename):
    def gen_sub_seq(l):
        l += 1
        for j in xrange(l):
            yield range(j + 1)

    for i in xrange(200):
        yield list(gen_sub_seq(i))
