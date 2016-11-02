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


# Define a py data provider
@provider(input_types=[
    sparse_vector(10000000),
    dense_vector(1)
])
def process(settings, filename):
    f = open(filename, 'r')
    for line in f:  # read each line
        splits = line.split(',')
        label = float(splits[0])
        splits.pop(0)
        sparse_float = []
        for value in splits:
            v = value.split(" ")
            sparse_float.append((long(v[0]), float(v[1])))
        # give data to paddle.
        yield sparse_float, [label]

    f.close()  # close file
