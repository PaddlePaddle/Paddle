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

from paddle.trainer.PyDataProvider2 import *
import gzip
import logging

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s', )
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)

OOV_POLICY_IGNORE = 0
OOV_POLICY_USE = 1
OOV_POLICY_ERROR = 2

num_original_columns = 3

# Feature combination patterns.
# [[-1,0], [0,0]]  means previous token at column 0 and current token at 
# column 0 are combined as one feature.
patterns = [
    [[-2, 0]],
    [[-1, 0]],
    [[0, 0]],
    [[1, 0]],
    [[2, 0]],
    [[-1, 0], [0, 0]],
    [[0, 0], [1, 0]],
    [[-2, 1]],
    [[-1, 1]],
    [[0, 1]],
    [[1, 1]],
    [[2, 1]],
    [[-2, 1], [-1, 1]],
    [[-1, 1], [0, 1]],
    [[0, 1], [1, 1]],
    [[1, 1], [2, 1]],
    [[-2, 1], [-1, 1], [0, 1]],
    [[-1, 1], [0, 1], [1, 1]],
    [[0, 1], [1, 1], [2, 1]],
]

dict_label = {
    'B-ADJP': 0,
    'I-ADJP': 1,
    'B-ADVP': 2,
    'I-ADVP': 3,
    'B-CONJP': 4,
    'I-CONJP': 5,
    'B-INTJ': 6,
    'I-INTJ': 7,
    'B-LST': 8,
    'I-LST': 9,
    'B-NP': 10,
    'I-NP': 11,
    'B-PP': 12,
    'I-PP': 13,
    'B-PRT': 14,
    'I-PRT': 15,
    'B-SBAR': 16,
    'I-SBAR': 17,
    'B-UCP': 18,
    'I-UCP': 19,
    'B-VP': 20,
    'I-VP': 21,
    'O': 22
}


def make_features(sequence):
    length = len(sequence)
    num_features = len(sequence[0])

    def get_features(pos):
        if pos < 0:
            return ['#B%s' % -pos] * num_features
        if pos >= length:
            return ['#E%s' % (pos - length + 1)] * num_features
        return sequence[pos]

    for i in xrange(length):
        for pattern in patterns:
            fname = '/'.join([get_features(i + pos)[f] for pos, f in pattern])
            sequence[i].append(fname)


'''
Source file format:
Each line is for one timestep. The features are separated by space.
An empty line indicates end of a sequence.

cutoff: a list of numbers. If count of a feature is smaller than this,
 it will be ignored.
if oov_policy[i] is OOV_POLICY_USE, id 0 is reserved for OOV features of
i-th column.

return a list of dict for each column
'''


def create_dictionaries(filename, cutoff, oov_policy):
    def add_to_dict(sequence, dicts):
        num_features = len(dicts)
        for features in sequence:
            l = len(features)
            assert l == num_features, "Wrong number of features " + line
            for i in xrange(l):
                if features[i] in dicts[i]:
                    dicts[i][features[i]] += 1
                else:
                    dicts[i][features[i]] = 1

    num_features = len(cutoff)
    dicts = []
    for i in xrange(num_features):
        dicts.append(dict())

    f = gzip.open(filename, 'rb')

    sequence = []

    for line in f:
        line = line.strip()
        if not line:
            make_features(sequence)
            add_to_dict(sequence, dicts)
            sequence = []
            continue
        features = line.split(' ')
        sequence.append(features)

    for i in xrange(num_features):
        dct = dicts[i]
        n = 1 if oov_policy[i] == OOV_POLICY_USE else 0
        todo = []
        for k, v in dct.iteritems():
            if v < cutoff[i]:
                todo.append(k)
            else:
                dct[k] = n
                n += 1

        if oov_policy[i] == OOV_POLICY_USE:
            # placeholder so that len(dct) will be the number of features
            # including OOV
            dct['#OOV#'] = 0

        logger.info('column %d dict size=%d, ignored %d' % (i, n, len(todo)))
        for k in todo:
            del dct[k]

    f.close()
    return dicts


def initializer(settings, **xargs):
    cutoff = [3, 1, 0]
    cutoff += [3] * len(patterns)
    oov_policy = [OOV_POLICY_IGNORE, OOV_POLICY_ERROR, OOV_POLICY_ERROR]
    oov_policy += [OOV_POLICY_IGNORE] * len(patterns)
    dicts = create_dictionaries('data/train.txt.gz', cutoff, oov_policy)
    dicts[2] = dict_label
    settings.dicts = dicts
    settings.oov_policy = oov_policy
    input_types = []
    num_features = len(dicts)
    for i in xrange(num_original_columns):
        input_types.append(integer_sequence(len(dicts[i])))
        logger.info("slot %s size=%s" % (i, len(dicts[i])))
    if patterns:
        dim = 0
        for i in xrange(num_original_columns, num_features):
            dim += len(dicts[i])
        input_types.append(sparse_binary_vector_sequence(dim))
        logger.info("feature size=%s" % dim)
    settings.input_types = input_types


'''
if oov_policy[i] == OOV_POLICY_USE, features in i-th column which are not
existed in dicts[i] will be assigned to id 0.
if oov_policy[i] == OOV_POLICY_ERROR, all features in i-th column MUST exist
in dicts[i].
'''


@provider(init_hook=initializer, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, filename):
    input_file = filename
    dicts = settings.dicts
    oov_policy = settings.oov_policy

    def gen_sample(sequence):
        num_features = len(dicts)
        sample = [list() for i in xrange(num_original_columns)]
        if patterns:
            sample.append([])
        for features in sequence:
            assert len(features) == num_features, \
                "Wrong number of features: " + line
            for i in xrange(num_original_columns):
                id = dicts[i].get(features[i], -1)
                if id != -1:
                    sample[i].append(id)
                elif oov_policy[i] == OOV_POLICY_IGNORE:
                    sample[i].append(0xffffffff)
                elif oov_policy[i] == OOV_POLICY_ERROR:
                    logger.fatal("Unknown token: %s" % features[i])
                else:
                    sample[i].append(0)

            if patterns:
                dim = 0
                vec = []
                for i in xrange(num_original_columns, num_features):
                    id = dicts[i].get(features[i], -1)
                    if id != -1:
                        vec.append(dim + id)
                    elif oov_policy[i] == OOV_POLICY_IGNORE:
                        pass
                    elif oov_policy[i] == OOV_POLICY_ERROR:
                        logger.fatal("Unknown token: %s" % features[i])
                    else:
                        vec.ids.append(dim + 0)

                    dim += len(dicts[i])
                sample[-1].append(vec)
        return sample

    num_features = len(dicts)
    f = gzip.open(input_file, 'rb')

    num_sequences = 0
    sequence = []
    for line in f:
        line = line.strip()
        if not line:
            make_features(sequence)
            yield gen_sample(sequence)
            sequence = []
            num_sequences += 1
            continue
        features = line.split(' ')
        sequence.append(features)

    f.close()

    logger.info("num_sequences=%s" % num_sequences)
