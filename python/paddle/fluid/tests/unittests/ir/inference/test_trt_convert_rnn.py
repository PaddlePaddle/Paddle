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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import os


class TrtConvertSliceTest(TrtLayerAutoScanTest):

    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824
        for hidden_size in [30]:
            for input_size in [30]:
                for batch in [2]:
                    for seq_len in [5]:
                        for num_layers in [1, 2]:
                            for is_bidirec in [True, False]:
                                dics = []
                                dics.append({
                                    "hidden_size": hidden_size,
                                    "input_size": input_size,
                                    "num_layers": num_layers,
                                    "mode": "LSTM",
                                    "is_bidirec": is_bidirec,
                                    "is_test": True,
                                    "dropout_prob": 0.0,
                                    # for my convience
                                    "batch": batch,
                                    "seq_len": seq_len,
                                })

                                K = 1
                                if (dics[0]["is_bidirec"]):
                                    K = 2

                                def generate_input1():
                                    return np.random.random([
                                        batch, seq_len, input_size
                                    ]).astype(np.float32) * 2 - 1

                                # initial input -> hidden
                                def generate_w0():
                                    return np.random.random([
                                        4 * hidden_size, input_size
                                    ]).astype(np.float32) * 2 - 1

                                # prev layer's output -> hidden
                                def generate_w1():
                                    return np.random.random([
                                        4 * hidden_size, K * hidden_size
                                    ]).astype(np.float32) * 2 - 1

                                #
                                def generate_w2():
                                    return np.random.random([
                                        4 * hidden_size, hidden_size
                                    ]).astype(np.float32) * 2 - 1

                                def generate_b():
                                    return np.random.random([
                                        4 * hidden_size
                                    ]).astype(np.float32) * 2 - 1

                                dics.append({
                                    "dtype":
                                    5,
                                    "input_dim_idx":
                                    0,
                                    "str_value":
                                    "",
                                    "value":
                                    0.0,
                                    "shape": [K * num_layers, -1, hidden_size],
                                    "output_dim_idx":
                                    1,
                                })
                                dics.append({"axis": [1, 0, 2]})
                                # set  weights
                                WeightList = [
                                    "weight" + str(i)
                                    for i in range(4 * K *
                                                   dics[0]["num_layers"])
                                ]
                                weights = {}
                                for i in range((int)(len(WeightList) / 2)):
                                    # mean this weight : input->hidden
                                    # input has 2 case: initial input input_size, K * hidden form the prev layer.
                                    if (i % 2 == 0):
                                        if (i <= K):
                                            weights[
                                                WeightList[i]] = TensorConfig(
                                                    data_gen=partial(
                                                        generate_w0))
                                        else:
                                            weights[
                                                WeightList[i]] = TensorConfig(
                                                    data_gen=partial(
                                                        generate_w1))
                                    # mean this weight : hidden->hidden
                                    if (i % 2 == 1):
                                        weights[WeightList[i]] = TensorConfig(
                                            data_gen=partial(generate_w2))
                                for i in range((int)(len(WeightList) / 2),
                                               len(WeightList)):
                                    weights[WeightList[i]] = TensorConfig(
                                        data_gen=partial(generate_b))
                                ops_config = [
                                    {
                                        "op_type":
                                        "fill_constant_batch_size_like",
                                        "op_inputs": {
                                            "Input": ["input_data"]
                                        },
                                        "op_outputs": {
                                            "Out": ["prestate1"]
                                        },
                                        "op_attrs": dics[1]
                                    },
                                    {
                                        "op_type":
                                        "fill_constant_batch_size_like",
                                        "op_inputs": {
                                            "Input": ["input_data"]
                                        },
                                        "op_outputs": {
                                            "Out": ["prestate2"]
                                        },
                                        "op_attrs": dics[1]
                                    },
                                    {
                                        "op_type": "transpose2",
                                        "op_inputs": {
                                            "X": ["input_data"]
                                        },
                                        "op_outputs": {
                                            "Out": ["rnn_input_data"]
                                        },
                                        "op_attrs": dics[2]
                                    },
                                    {
                                        "op_type": "rnn",
                                        "op_inputs": {
                                            "Input": ["rnn_input_data"],
                                            # prev_c, prev_h
                                            "PreState":
                                            ["prestate1", "prestate2"],
                                            "WeightList": WeightList,
                                        },
                                        "op_outputs": {
                                            "Out": ["rnn_output_data"],
                                            "State": [
                                                "state_output_data0",
                                                "state_output_data1"
                                            ],
                                            "Reserve": ["reserve_data"],
                                            "DropoutState":
                                            ["DropoutState_data"]
                                        },
                                        "op_attrs": dics[0]
                                    }
                                ]
                                ops = self.generate_op_config(ops_config)

                                program_config = ProgramConfig(
                                    ops=ops,
                                    weights=weights,
                                    inputs={
                                        "input_data":
                                        TensorConfig(
                                            data_gen=partial(generate_input1))
                                    },
                                    outputs=["rnn_output_data"])

                                yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]
        num_layers = attrs[3]["num_layers"]
        hidden_size = attrs[3]["hidden_size"]
        batch = attrs[3]["batch"]
        input_size = attrs[3]["input_size"]
        seq_len = attrs[3]["seq_len"]

        K = 1
        if attrs[3]["is_bidirec"]:
            K = 2

        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [batch - 1, seq_len, input_size],
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [batch + 1, seq_len, input_size],
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [batch, seq_len, input_size],
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs for i in range(len(program_config.ops))
        ]

        # The output has diff between gpu and trt in PR-CI-Windows-Inference
        tol_fp32 = 1e-5
        tol_half = 1e-2
        if (os.name == 'nt'):
            tol_fp32 = 1e-2
            tol_half = 1e-1

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), tol_fp32
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), tol_half

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
