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

import sys
sys.path.append("../")

from paddle.trainer.PyDataProviderWrapper import *
import random
import json
import string

SPARSE_ID_LIMIT = 1000
SPARSE_ID_COUNT = 100
SEQUENCE_LIMIT = 50
STRING_LIMIT = 10

sparse_id_randomer = lambda: random.randrange(0, SPARSE_ID_LIMIT - 1)
sparse_count_randomer = lambda: random.randrange(1, SPARSE_ID_COUNT)
val_randomer = lambda: random.uniform(-1.0, 1.0)
seq_count_randomer = lambda: random.randrange(1, SEQUENCE_LIMIT)
str_count_randomer = lambda: random.randrange(1, STRING_LIMIT)


class IDRandomer():  # A random generator, return unique id
    def __init__(self):
        self.id_set = set()

    def __call__(self):
        idx = sparse_id_randomer()
        if idx not in self.id_set:
            self.id_set.add(idx)
            return idx
        else:
            return self.__call__()


# SparseValueSlot
def sparse_value_creator(_):
    rand = IDRandomer()
    return [(rand(), val_randomer()) for _ in xrange(sparse_count_randomer())]


sparse_value = map(sparse_value_creator, range(seq_count_randomer()))


# DenseSlot
def dense_creator(_):
    return [val_randomer() for _ in xrange(SPARSE_ID_LIMIT)]


dense = map(dense_creator, range(seq_count_randomer()))


# SparseNonValueSlot
def sparse_creator(_):
    rand = IDRandomer()
    return [rand() for _ in xrange(sparse_count_randomer())]


sparse_nonvalue = map(sparse_creator, range(seq_count_randomer()))

# IndexSlot
ids = [sparse_id_randomer() for _ in range(seq_count_randomer())]


# StringSlot
def random_str(size=8, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


strs = [random_str(str_count_randomer()) for _ in range(seq_count_randomer())]


def processSeqAndGenerateDataInit(obj, *args, **kwargs):
    obj.json_filename = kwargs.get("load_data_args", "test_data.json")


@provider(
    slots=[
        SparseValueSlot(SPARSE_ID_LIMIT), DenseSlot(SPARSE_ID_LIMIT),
        SparseNonValueSlot(SPARSE_ID_LIMIT), IndexSlot(SPARSE_ID_LIMIT),
        StringSlot(SPARSE_ID_LIMIT)
    ],
    use_seq=True,
    init_hook=processSeqAndGenerateDataInit)
def processSeqAndGenerateData(obj, name):
    retv = [sparse_value, dense, sparse_nonvalue, ids, strs]
    # Write to protoseq.
    with open(obj.json_filename, "w") as f:
        json.dump(retv, f)
    yield retv


def processSubSeqAndGenerateDataInit(obj, *args, **kwargs):
    obj.json_filename = kwargs.get("load_data_args", "test_data.json")


@provider(
    slots=[
        SparseValueSlot(SPARSE_ID_LIMIT), DenseSlot(SPARSE_ID_LIMIT),
        SparseNonValueSlot(SPARSE_ID_LIMIT), IndexSlot(SPARSE_ID_LIMIT),
        StringSlot(SPARSE_ID_LIMIT)
    ],
    use_seq=True,
    init_hook=processSubSeqAndGenerateDataInit)
def processSubSeqAndGenerateData(obj, name):
    retv_json = [sparse_value, dense, sparse_nonvalue, ids, strs]
    retv_wrapper = [[sparse_value], [dense], [sparse_nonvalue], [ids], [strs]]
    # Write to protoseq.
    with open(obj.json_filename, "w") as f:
        json.dump(retv_json, f)
    yield retv_wrapper


if __name__ == "__main__":
    pvd = processSeqAndGenerateData("_")
    print pvd.getNextBatch(100)
    pvd = processSubSeqAndGenerateData("_")
    print pvd.getNextBatch(1)
