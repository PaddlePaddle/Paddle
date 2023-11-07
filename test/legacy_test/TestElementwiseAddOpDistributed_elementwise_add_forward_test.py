# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import pathlib
import pickle
import sys

sys.path.append(
    str(pathlib.Path(__file__).resolve().parents[0] / 'test/legacy_test')
)
from auto_parallel_op_test import AutoParallelForwardChecker

file = open(
    "./TestElementwiseAddOpDistributed_elementwise_add_forward_info.pkl", "rb"
)
test_info = pickle.load(file)
print("check auto parallel forward.............")
auto_parallel_forward_checker = AutoParallelForwardChecker(
    test_info["op_type"],
    test_info["python_api"],
    test_info["dtype"],
    test_info["input_specs"],
    test_info["inputs"],
    test_info["attrs"],
    test_info["outputs"],
    test_info["place"],
    test_info["python_out_sig"],
)
auto_parallel_forward_checker.check()
print("check auto parallel forward success!")
