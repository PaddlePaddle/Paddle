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
"""Load checkpoint of running LTR ranking to do prediction and save inference model."""

import os
import sys
import time
import numpy as np
import paddle
print(paddle.__file__)
import paddle.fluid as fluid

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType

from paddle.inference import InternalUtils

###origin_model_file = "/root/auto_compress/NLP/ernie-3.0/all_original_models/AFQMC/infer.pdmodel"
###origin_params_file = "/root/auto_compress/NLP/ernie-3.0/all_original_models/AFQMC/infer.pdiparams"
# int8_model_file = "./old_format_quant_fc_matmul_add/infer.pdmodel"
# int8_params_file = "./old_format_quant_fc_matmul_add/infer.pdiparams"
# model_dir = "./ernie_model_4/"
# ernie模型 有这种 model pdiparams 模式的吗

#int8_model_file = "./fake_quant_fc/infer.pdmodel"
#int8_params_file = "./fake_quant_fc/infer.pdiparams"
def init_predictor(precision, model_dir): 
    config = Config(model_dir)
    # config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(100, 0)
    precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
    }
    # precision = "fp16"
#    config.exp_disable_tensorrt_ops(["elementwise_sub"])
    config.enable_tensorrt_engine(workspace_size=1 << 30 - 1,
                                  max_batch_size=1,
                                  min_subgraph_size=5,
                                  precision_mode=precision_map[precision],
                                  use_static=False,
                                  use_calib_mode=False)
    names = [
        "read_file_0.tmp_0",
        "read_file_0.tmp_1",
        "read_file_0.tmp_2",
        "read_file_0.tmp_4"

    ]
    max_batch = 32
    max_single_seq_len = 128
    opt_single_seq_len = 64
    min_batch_seq_len = 1
    max_batch_seq_len = 512
    opt_batch_seq_len = 256

    # min_shape = [min_batch_seq_len, min_batch_seq_len]
    # max_shape = [max_batch_seq_len, max_batch_seq_len]
    # opt_shape = [opt_batch_seq_len, opt_batch_seq_len]

    min_shape = [min_batch_seq_len]
    max_shape = [max_batch_seq_len]
    opt_shape = [opt_batch_seq_len]

    config.set_trt_dynamic_shape_info(
    {
        names[0]: min_shape,
        names[1]: min_shape,
        names[2]: [1],
        names[3]: [1, 1, 1],
        
    },{
        names[0]: max_shape,
        names[1]: max_shape,
        names[2]: max_shape,
        names[3]: [1, max_single_seq_len, 1],
        
    },{
        names[0]: opt_shape,
        names[1]: opt_shape,
        names[2]: [max_batch + 1],
        names[3]: [1, opt_single_seq_len, 1],
        
    })

    # config.enable_tensorrt_oss()
    config.enable_tensorrt_varseqlen()
    
    # print(names[1])
    InternalUtils.set_transformer_posid(config,names[2])
    InternalUtils.set_transformer_maskid(config,names[3])
    # print(333)
    predictor = create_predictor(config)
    # print(444)
    return predictor

def run(predictor, data):
    input_names = predictor.get_input_names()
    
    for i, name in enumerate(input_names):
        # print(i, name)
        input_tensor = predictor.get_input_handle(name)
        # print('input_tensor', input_tensor.shape)
        # 设置tensor的维度信息适应输入数据
        input_tensor.reshape(data[i].shape)
        # print(data[i].shape)
        # 从模型获取CPU 设置到tensor内部
        input_tensor.copy_from_cpu(data[i].copy())
    # print("do the inference")
    # do the inference
    predictor.run()
    # print("do results!!!")
    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        # 从tensor中获取数据到CPU
        output_data = output_tensor.copy_to_cpu()
        # print(output_data)
        results.append(output_data)

    return results


def main(precision, model_dir):
    # print(2222)
    paddle.enable_static()
    predictor = init_predictor(precision, model_dir)
    # print(3333)
    run_batch = 2
    run_seq_len = 71
    max_seq_len = 128
    src_ids = np.array([1,
      3558,
      4,
      75,
      491,
      89,
      340,
      313,
      93,
      4,
      255,
      10,
      75,
      321,
      4095,
      1902,
      4,
      134,
      49,
      75,
      311,
      14,
      44,
      178,
      543,
      15,
      12043,
      2,
      75,
      201,
      340,
      9,
      14,
      44,
      486,
      218,
      1140,
      279,
      12043,
      2,
    #   // sentence 2
      101,
      2054,
      2234,
      2046,
      2486,
      2044,
      1996,
      2047,
      4552,
      2001,
      9536,
      1029,
      102,
      2004,
      1997,
      2008,
      2154,
      1010,
      1996,
      2047,
      4552,
      9536,
      2075,
      1996,
      2117,
      3072,
      2234,
      2046,
      2486,
      1012,
      102], dtype=np.int64)
    src_ids2 = np.array([0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                            #  // sentence 2
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             0,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1], dtype=np.int64)
    sent_ids = np.array([0, 40, 71], dtype=np.int64)
    sent_ids2 = np.zeros((1, max_seq_len, 1), dtype=np.float32)


    # src_ids = np.random.randint(1, 200, seq_len, dtype=np.int64)
    # src_ids2 = np.random.randint(1, 2, seq_len, dtype=np.int64)
    # sent_ids = np.random.randint(1, 2, seq_len, dtype=np.int64)
    # sent_ids2 = np.zeros((1, 16, 1), dtype=np.float32)

    # src_ids = ([], dtype=np.int64)
    # src_ids2 = np.random.randint([], dtype=np.int64)
    # sent_ids = np.random.randint([0, 40, 71], dtype=np.int64)
    # sent_ids2 = np.zeros(128, dtype=np.float32)

    # print(src_ids)
    run_data = [src_ids, src_ids2, sent_ids, sent_ids2]
    # print(run_data)
    for _ in range(100):
        np_probs = run(predictor, run_data)

    s_time = time.time()
    for _ in range(1000):
        np_probs = run(predictor, run_data)
    print(np_probs)
    print(f"time per batch: {1000*(time.time() - s_time)/1000}ms")


if __name__ == '__main__':
    # print(1111)
    # batch_size = int(sys.argv[1])
    # seq_len = int(sys.argv[2])
    precision = sys.argv[1]
    model_dir = sys.argv[2]
    # pruning = sys.argv[4]
    # pruning = True if pruning == "true" else False
    # main(batch_size, seq_len, precision, pruning)
    main(precision, model_dir)


