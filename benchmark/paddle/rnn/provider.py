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

import io, os
import random
import numpy as np
import six.moves.cPickle as pickle
from paddle.trainer.PyDataProvider2 import *


def remove_unk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]


# ==============================================================
#  tensorflow uses fixed length, but PaddlePaddle can process
#  variable-length. Padding is used in benchmark in order to
#  compare with other platform. 
# ==============================================================
def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='post',
                  truncating='post',
                  value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def initHook(settings, vocab_size, pad_seq, maxlen, **kwargs):
    settings.vocab_size = vocab_size
    settings.pad_seq = pad_seq
    settings.maxlen = maxlen
    settings.input_types = [
        integer_value_sequence(vocab_size), integer_value(2)
    ]


@provider(
    init_hook=initHook, min_pool_size=-1, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file):
    f = open(file, 'rb')
    train_set = pickle.load(f)
    f.close()
    x, y = train_set

    # remove unk, namely remove the words out of dictionary
    x = remove_unk(x, settings.vocab_size)
    if settings.pad_seq:
        x = pad_sequences(x, maxlen=settings.maxlen, value=0.)

    for i in range(len(y)):
        yield map(int, x[i]), int(y[i])
