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

DICT_DIM = 3000


@provider(input_types=[integer_sequence(DICT_DIM), integer_value(DICT_DIM)])
def process(settings, filename):
    with open(filename) as f:
        # yield word ids to predict inner word id
        # such as [28, 29, 10, 4], 4
        # It means the sentance is  28, 29, 4, 10, 4.
        yield read_next_from_file(f)
