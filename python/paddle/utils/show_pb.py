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
Show the content of proto buffer data file of PADDLE
"""

import os
import sys
from google.protobuf.internal.decoder import _DecodeVarint
import paddle.proto.DataFormat_pb2 as DataFormat


def read_proto(file, message):
    """
    read a protobuffer struct from file, the length of the struct is stored as
    a varint, then followed by the actual struct data.
    @return True success, False for end of file
    """

    buf = file.read(8)
    if not buf:
        return False
    result, pos = _DecodeVarint(buf, 0)
    buf = buf[pos:] + file.read(result - len(buf) + pos)
    message.ParseFromString(buf)

    return True


def usage():
    print >> sys.stderr, "Usage: python show_pb.py PROTO_DATA_FILE"
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()

    f = open(sys.argv[1])
    header = DataFormat.DataHeader()
    read_proto(f, header)
    print header

    sample = DataFormat.DataSample()
    while read_proto(f, sample):
        print sample
