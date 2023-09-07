# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
import time

import numpy as np

import paddle.inference as paddle_infer
from paddle.base.core import AnalysisConfig, create_paddle_predictor


def main():
    args = parse_args()

    config = set_config(args)

    predictor = create_paddle_predictor(config)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])
    fake_input = np.random.randn(1, 3, 224, 224).astype("float32")
    input_tensor.reshape([1, 3, 224, 224])
    input_tensor.copy_from_cpu(fake_input)

    if len(input_names) > 1:
        input_tensor2 = predictor.get_input_tensor(input_names[1])
        fake_input2 = np.random.randn(512, 512).astype("float32")
        input_tensor2.reshape([512, 512])
        input_tensor2.copy_from_cpu(fake_input2)

    for _ in range(0, 10):
        predictor.zero_copy_run()

    time1 = time.time()
    repeat = 10
    for i in range(0, repeat):
        predictor.zero_copy_run()
    time2 = time.time()
    total_inference_cost = (time2 - time1) * 1000  # total time cost(ms)
    print(f"Average latency : {total_inference_cost / repeat} ms")
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])
    output_data = output_tensor.copy_to_cpu()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="model filename")

    return parser.parse_args()


def set_config(args):
    config = AnalysisConfig(
        args.model_dir + '/__model__', args.model_dir + '/params'
    )
    config.enable_profile()
    config.enable_use_gpu(1000, 1)
    # Enable TensorRT
    config.enable_tensorrt_engine(
        workspace_size=1 << 30,
        max_batch_size=1,
        min_subgraph_size=3,
        precision_mode=paddle_infer.PrecisionType.Float32,
        use_static=False,
        use_calib_mode=False,
    )
    config.enable_memory_optim()
    config.gpu_device_id()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.switch_ir_optim(True)
    # To test cpu backend, just uncomment the following 2 lines.
    # config.switch_ir_optim(True)
    # config.disable_gpu()
    # config.enable_mkldnn()
    return config


if __name__ == "__main__":
    main()
