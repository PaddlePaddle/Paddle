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

import argparse
import paddle
import pickle
import importlib
import os
import sys
from paddle.distributed.fleet.launch_utils import run_with_coverage
from dist_pass_test_base import prepare_python_path_and_return_module, DistPassTestBase


def parse_args():
    parser = argparse.ArgumentParser(
        description='arguments for distributed pass tests')
    parser.add_argument('--file_path', type=str, help='The test file path.')
    parser.add_argument(
        '--class_name',
        type=str,
        help='The test class name. It is the class name that inherits the DistPassTestBase class.'
    )
    parser.add_argument(
        '--apply_pass',
        default=False,
        action="store_true",
        help='Whether to apply distributed passes.')
    parser.add_argument(
        '--input_file',
        type=str,
        help='The input file which contains the dumped input arguments.')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='The output directory to save the logs and output results.')
    parser.add_argument(
        '--model_file',
        type=str,
        help='The input model file which contains the dumped model function.')
    return parser.parse_args()


def run_main(args):
    if os.environ.get("WITH_COVERAGE", "OFF") == "ON":
        run_with_coverage(True)
    module_name = prepare_python_path_and_return_module(args.file_path)
    test_module = importlib.import_module(module_name)
    test_class = getattr(test_module, args.class_name)
    assert issubclass(test_class, DistPassTestBase)
    test_obj = test_class()
    rank = paddle.distributed.get_rank()
    with open(args.input_file, "rb") as f:
        kwargs = pickle.load(f)

    output_file = "{}/{}.bin".format(args.output_dir, rank)
    if args.model_file:
        with open(args.model_file, "rb") as f:
            model = pickle.load(f)
    else:
        model = None

    try:
        test_obj.setUpClass()
        test_obj.setUp()
        test_obj._run_gpu_main(model, args.apply_pass, output_file, **kwargs)
    finally:
        test_obj.tearDown()
        test_obj.tearDownClass()


if __name__ == "__main__":
    args = parse_args()
    run_main(args)
