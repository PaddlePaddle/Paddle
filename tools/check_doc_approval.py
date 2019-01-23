# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import ast
import hashlib
import importlib
import collections
import paddle.fluid

files = [
    "paddle.fluid", "paddle.fluid.average", "paddle.fluid.backward",
    "paddle.fluid.clip", "paddle.fluid.data_feeder", "paddle.fluid.executor",
    "paddle.fluid.initializer", "paddle.fluid.io", "paddle.fluid.layers",
    "paddle.fluid.metrics", "paddle.fluid.nets", "paddle.fluid.optimizer",
    "paddle.fluid.profiler", "paddle.fluid.recordio_writer",
    "paddle.fluid.regularizer", "paddle.fluid.transpiler"
]


def md5(doc):
    hash = hashlib.md5()
    hash.update(str(doc))
    return hash.hexdigest()


def get_module():
    for fi in files:
        fi_lib = importlib.import_module(fi)
        doc_function = getattr(fi_lib, "__all__")
        for api in doc_function:
            api_name = fi + "." + api
            try:
                doc_module = getattr(eval(api_name), "__doc__")
            except:
                pass
            doc_md5_code = md5(doc_module)
            doc_dict[api_name] = doc_md5_code


if __name__ == "__main__":
    doc_dict = collections.OrderedDict()
    get_module()
    print(doc_dict)
