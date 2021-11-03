# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import pickle
import importlib
import os
import sys
from paddle.distributed.fleet.launch_utils import run_with_coverage


def run_main(module_name, test_class_name, apply_pass, input_file,
             output_file_prefix):
    if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
        run_with_coverage(True)
    assert apply_pass.lower() in ['true', 'false']
    apply_pass = (apply_pass.lower() == 'true')
    module = importlib.import_module(module_name)
    test_class = getattr(module, test_class_name)
    test_obj = test_class()
    rank = paddle.distributed.get_rank()
    with open(input_file, "rb") as f:
        kwargs = pickle.load(f)

    output_file = "{}/{}.bin".format(output_file_prefix, rank)

    try:
        test_obj.setUpClass()
        test_obj.setUp()
        test_obj._run_gpu_main(apply_pass, output_file, **kwargs)
    finally:
        test_obj.tearDown()
        test_obj.tearDownClass()


if __name__ == "__main__":
    run_main(*sys.argv[1:])
