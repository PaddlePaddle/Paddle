# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from paddle.inference import Config, PrecisionType, create_predictor


def init_predictor(args):
    if args.model_dir != "":
        config = Config(args.model_dir)
    else:
        config = Config(args.model_file, args.params_file)

    config.enable_memory_optim()

    gpu_precision = PrecisionType.Float32
    if args.run_mode == "gpu_fp16":
        gpu_precision = PrecisionType.Half

    config.enable_use_gpu(1000, 0, gpu_precision)
    # config.switch_ir_optim(False)

    config.enable_new_executor()
    #config.enable_new_ir()

    predictor = create_predictor(config)
    return predictor


def run(predictor):
    input_names = predictor.get_input_names()
    x = np.ones([16, 16], np.float32)
    y = np.ones([16, 16], np.float32)

    input_tensor_0 = predictor.get_input_handle(input_names[0])
    input_tensor_0.reshape(x.shape)
    input_tensor_0.copy_from_cpu(x)

    input_tensor_1 = predictor.get_input_handle(input_names[1])
    input_tensor_1.reshape(y.shape)
    input_tensor_1.copy_from_cpu(y)

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        output_data = output_tensor.copy_to_cpu()
        results.append(output_data)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Model filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        default="",
        help="Parameter filename, Specify this when your model is a combined model.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Model dir, If you load a non-combined model, specify the directory of the model.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="",
        help="Run_mode which can be: trt_fp32, trt_fp16, trt_int8 and gpu_fp16.",
    )
    parser.add_argument(
        "--use_dynamic_shape",
        type=int,
        default=0,
        help="Whether use trt dynamic shape.",
    )
    parser.add_argument(
        "--use_collect_shape",
        type=int,
        default=0,
        help="Whether use trt collect shape.",
    )
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="The file path of dynamic shape info.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred = init_predictor(args)

    result = run(pred)
    print("class index: ", result)
