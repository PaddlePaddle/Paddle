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

import random

import numpy as np
from py_paddle import swig_paddle


def doubleEqual(a, b):
    return abs(a - b) < 1e-5


def __readFromFile():
    for i in xrange(10002):
        label = np.random.randint(0, 9)
        sample = np.random.rand(784) + 0.1 * label
        yield sample, label


def loadMNISTTrainData(batch_size=100):
    if not hasattr(loadMNISTTrainData, "gen"):
        generator = __readFromFile()
        loadMNISTTrainData.gen = generator
    else:
        generator = loadMNISTTrainData.gen
    args = swig_paddle.Arguments.createArguments(2)
    # batch_size = 100

    dense_slot = []
    id_slot = []
    atEnd = False

    for _ in xrange(batch_size):
        try:
            result = generator.next()
            dense_slot.extend(result[0])
            id_slot.append(result[1])
        except StopIteration:
            atEnd = True
            del loadMNISTTrainData.gen
            break

    dense_slot = swig_paddle.Matrix.createDense(dense_slot, batch_size, 784)
    id_slot = swig_paddle.IVector.create(id_slot)
    args.setSlotValue(0, dense_slot)
    args.setSlotIds(1, id_slot)
    return args, atEnd
