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

from cStringIO import StringIO

import paddle.proto.DataFormat_pb2 as DataFormat
from google.protobuf.internal.encoder import _EncodeVarint

import logging
import pprint

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

    f = open(filename, 'rb')

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


def encode_varint(v):
    out = StringIO()
    _EncodeVarint(out.write, v)
    return out.getvalue()


def write_proto(file, message):
    s = message.SerializeToString()
    packed_len = encode_varint(len(s))
    file.write(packed_len + s)


'''
if oov_policy[i] == OOV_POLICY_USE, features in i-th column which are not
existed in dicts[i] will be assigned to id 0.
if oov_policy[i] == OOV_POLICY_ERROR, all features in i-th column MUST exist
in dicts[i].
'''


def gen_proto_file(input_file, dicts, oov_policy, output_file):
    def write_sequence(out, sequence):
        num_features = len(dicts)
        is_beginning = True
        for features in sequence:
            assert len(features) == num_features, \
                "Wrong number of features: " + line
            sample = DataFormat.DataSample()
            for i in xrange(num_original_columns):
                id = dicts[i].get(features[i], -1)
                if id != -1:
                    sample.id_slots.append(id)
                elif oov_policy[i] == OOV_POLICY_IGNORE:
                    sample.id_slots.append(0xffffffff)
                elif oov_policy[i] == OOV_POLICY_ERROR:
                    logger.fatal("Unknown token: %s" % features[i])
                else:
                    sample.id_slots.append(0)

            if patterns:
                dim = 0
                vec = sample.vector_slots.add()
                for i in xrange(num_original_columns, num_features):
                    id = dicts[i].get(features[i], -1)
                    if id != -1:
                        vec.ids.append(dim + id)
                    elif oov_policy[i] == OOV_POLICY_IGNORE:
                        pass
                    elif oov_policy[i] == OOV_POLICY_ERROR:
                        logger.fatal("Unknown token: %s" % features[i])
                    else:
                        vec.ids.append(dim + 0)

                    dim += len(dicts[i])

            sample.is_beginning = is_beginning
            is_beginning = False
            write_proto(out, sample)

    num_features = len(dicts)
    f = open(input_file, 'rb')
    out = open(output_file, 'wb')

    header = DataFormat.DataHeader()
    if patterns:
        slot_def = header.slot_defs.add()
        slot_def.type = DataFormat.SlotDef.VECTOR_SPARSE_NON_VALUE
        slot_def.dim = sum(
            [len(dicts[i]) for i in xrange(num_original_columns, len(dicts))])
        logger.info("feature_dim=%s" % slot_def.dim)

    for i in xrange(num_original_columns):
        slot_def = header.slot_defs.add()
        slot_def.type = DataFormat.SlotDef.INDEX
        slot_def.dim = len(dicts[i])

    write_proto(out, header)

    num_sequences = 0
    sequence = []
    for line in f:
        line = line.strip()
        if not line:
            make_features(sequence)
            write_sequence(out, sequence)
            sequence = []
            num_sequences += 1
            continue
        features = line.split(' ')
        sequence.append(features)

    f.close()
    out.close()

    logger.info("num_sequences=%s" % num_sequences)


dict2 = {
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

if __name__ == '__main__':
    cutoff = [3, 1, 0]
    cutoff += [3] * len(patterns)
    oov_policy = [OOV_POLICY_IGNORE, OOV_POLICY_ERROR, OOV_POLICY_ERROR]
    oov_policy += [OOV_POLICY_IGNORE] * len(patterns)
    dicts = create_dictionaries('trainer/tests/train.txt', cutoff, oov_policy)
    dicts[2] = dict2
    gen_proto_file('trainer/tests/train.txt', dicts, oov_policy,
                   'trainer/tests/train_proto.bin')
    gen_proto_file('trainer/tests/test.txt', dicts, oov_policy,
                   'trainer/tests/test_proto.bin')
