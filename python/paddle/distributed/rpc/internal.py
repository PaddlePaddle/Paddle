# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import pickle
from collections import namedtuple

PythonFunc = namedtuple("PythonFunc", ["func", "args", "kwargs"])
"""Some Python code interfaces called in C++"""


def _serialize(obj):
    return pickle.dumps(obj)


def _deserialize(obj):
    return pickle.loads(obj)


def _run_py_func(python_func):
    result = python_func.func(*python_func.args, **python_func.kwargs)
    return result
