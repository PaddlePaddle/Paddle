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

import paddle.v2 as paddle
import tarfile
import os
import pickle

SPLIT_COUNT = 3
N = 5


def file_len(fd):
    for i, l in enumerate(fd):
        pass
    return i + 1


def split_from_reader_by_line(filename, reader, split_count):
    fn = open(filename, "w")
    for batch_id, batch_data in enumerate(reader()):
        batch_data_str = [str(d) for d in batch_data]
        fn.write(",".join(batch_data_str))
        fn.write("\n")
    fn.close()

    fn = open(filename, "r")
    total_line_count = file_len(fn)
    fn.close()
    per_file_lines = total_line_count / split_count + 1
    cmd = "split -d -a 5 -l %d %s %s-" % (per_file_lines, filename, filename)
    os.system(cmd)


word_dict = paddle.dataset.imikolov.build_dict()
with open("word_dict.pickle", "w") as dict_f:
    pickle.dump(word_dict, dict_f)

split_from_reader_by_line("train.txt",
                          paddle.dataset.imikolov.train(word_dict, N),
                          SPLIT_COUNT)
split_from_reader_by_line("test.txt",
                          paddle.dataset.imikolov.test(word_dict, N),
                          SPLIT_COUNT)
