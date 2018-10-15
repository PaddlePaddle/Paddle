#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy
import struct
import traceback


def header_creator():
    ret = ""
    ret += struct.pack('i', 3)  # slot num
    ret += struct.pack('i', 1)  # sequence flag
    ret += struct.pack('i', 0)  # slot0 dense type
    ret += struct.pack('i', 3)  # slot0 dim
    ret += struct.pack('i', 1)  # slot1 sparse non value type
    ret += struct.pack('i', 7)  # slot1 dim
    ret += struct.pack('i', 3)  # slot2 index type
    ret += struct.pack('i', 2)  # slot2 dim
    return ret


def dense_value_creator(sample_num):
    ret = ""
    ret += struct.pack('i', sample_num)  # slot0 sample num
    for i in range(sample_num):  # slot0 value
        ret += struct.pack('f', 1.0)
        ret += struct.pack('f', 2.0)
        ret += struct.pack('f', 3.0)
    return ret


def sparse_value_creator(sample_num):
    ret = ""
    ret += struct.pack('i', sample_num)  # slot1 sample num
    for i in range(sample_num):  # slot1 index
        ret += struct.pack('i', i * 2)
    ret += struct.pack('i', sample_num * 2)  #slot1 length
    for i in range(sample_num):  # slot1 value
        ret += struct.pack('i', 1)
        ret += struct.pack('i', 2)
    return ret


def index_value_creator(sample_num):
    ret = ""
    ret += struct.pack('i', sample_num)  # slot2 sample num
    for i in range(sample_num):  # slot2 value
        ret += struct.pack('i', 0)
    return ret


def sequenceStartPositions_creator():
    ret = ""
    ret += struct.pack('i', 2)  # slot0 sequence num
    ret += struct.pack('i', 0)  # slot0 sequence value1
    ret += struct.pack('i', 1)  # slot0 sequence value2
    ret += struct.pack('i', 1)  # slot1 sequence num
    ret += struct.pack('i', 0)  # slot1 sequence value1
    ret += struct.pack('i', 2)  # slot2 sequence num
    ret += struct.pack('i', 0)  # slot2 sequence value1
    ret += struct.pack('i', 1)  # slot2 sequence value2
    return ret


def subSequenceStartPositions_creator():
    ret = ""
    ret += struct.pack('i', 3)  # slot0 subsequence num
    ret += struct.pack('i', 0)  # slot0 subsequence value1
    ret += struct.pack('i', 1)  # slot0 subsequence value2
    ret += struct.pack('i', 2)  # slot0 subsequence value3
    ret += struct.pack('i', 2)  # slot1 subsequence num
    ret += struct.pack('i', 0)  # slot1 subsequence value1
    ret += struct.pack('i', 1)  # slot1 subsequence value2
    ret += struct.pack('i', 3)  # slot2 subsequence num
    ret += struct.pack('i', 0)  # slot2 subsequence value1
    ret += struct.pack('i', 1)  # slot2 subsequence value2
    ret += struct.pack('i', 2)  # slot2 subsequence value3
    return ret


class SimpleDataProvider:
    def __init__(self, *file_list):
        self.file_list = file_list

    def shuffle(self):
        pass

    def reset(self):
        pass

    def getHeader(self):
        return header_creator()

    def getNextBatch(self, batch_size):
        ret = ""
        ret += struct.pack('i', 2)  # batch size
        ret += dense_value_creator(2)  # slot0
        ret += sparse_value_creator(2)  # slot1
        ret += index_value_creator(2)  # slot2
        ret += sequenceStartPositions_creator()
        return ret


class SimpleNestDataProvider:
    def __init__(self, *file_list):
        self.file_list = file_list

    def shuffle(self):
        pass

    def reset(self):
        pass

    def getHeader(self):
        return header_creator()

    def getNextBatch(self, batch_size):
        ret = ""
        ret += struct.pack('i', 2)  # batch size
        ret += dense_value_creator(4)  # slot0
        ret += sparse_value_creator(4)  # slot1
        ret += index_value_creator(4)  # slot2
        ret += sequenceStartPositions_creator()
        ret += subSequenceStartPositions_creator()
        return ret


if __name__ == "__main__":
    # test code
    data_provider = SimpleDataProvider('./test_batch')
    print len(data_provider.getHeader())
    print len(data_provider.getNextBatch(2))

    data_provider = SimpleNestDataProvider('./test_batch')
    print len(data_provider.getHeader())
    print len(data_provider.getNextBatch(2))
