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


def doc_md5_dict(doc_md5_path):
    with open(doc_md5_path, "rb") as f:
        doc_md5 = f.read()
        doc_md5_dict = ast.literal_eval(doc_md5)
    return doc_md5_dict


def check_doc_md5():
    for k, v in doc_dict.items():
        try:
            if doc_ci_dict[k] != v:
                return doc_dict
        except:
            return doc_dict
    return True


if __name__ == "__main__":
    doc_dict = {}
    doc_ci_dict = {}
    doc_md5_file = "/root/.cache/doc_md5.txt"
    if not os.path.exists(doc_md5_file):
        os.mknod(doc_md5_file)
    else:
        doc_ci_dict = doc_md5_dict(doc_md5_file)
    get_module()
    if not os.path.getsize(doc_md5_file):
        with open(doc_md5_file, 'w') as f:
            f.write(str(doc_dict))
        check_dic = True
        print(check_dic)
    else:
        check_dic = check_doc_md5()
        print(check_dic)
