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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TrtConvertYoloBoxTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self):
        def generate_input1(attrs: List[Dict[str, Any]], batch, channel):
            if attrs[0]['iou_aware'] == True:
                return np.ones(
                    [batch, 3 * (channel + 6), 13, 13]).astype(np.float32)
            else:
                return np.ones(
                    [batch, 3 * (channel + 5), 13, 13]).astype(np.float32)

        def generate_input2(attrs: List[Dict[str, Any]], batch):
            return np.random.random([batch, 2]).astype(np.int32)

        for batch in [1, 4]:
            for class_num in [80, 30]:
                for anchors in [[10, 13, 16, 30, 33, 23]]:
                    for downsample_ratio in [32, 16]:
                        for conf_thresh in [0.01, 0.02]:
                            for clip_bbox in [True, False]:
                                for scale_x_y in [1.0, 0.9]:
                                    for iou_aware in [False, True]:
                                        for iou_aware_factor in [0.5]:
                                            dics = [{
                                                "class_num": class_num,
                                                "anchors": anchors,
                                                "downsample_ratio":
                                                downsample_ratio,
                                                "conf_thresh": conf_thresh,
                                                "clip_bbox": clip_bbox,
                                                "scale_x_y": scale_x_y,
                                                "iou_aware": iou_aware,
                                                "iou_aware_factor":
                                                iou_aware_factor
                                            }, {}]
                                            ops_config = [{
                                                "op_type": "yolo_box",
                                                "op_inputs": {
                                                    "X": ["yolo_box_input"],
                                                    "ImgSize": ["imgsize"]
                                                },
                                                "op_outputs": {
                                                    "Boxes": ["boxes"],
                                                    "Scores": ["scores"]
                                                },
                                                "op_attrs": dics[0]
                                            }]
                                            ops = self.generate_op_config(
                                                ops_config)
                                            program_config = ProgramConfig(
                                                ops=ops,
                                                weights={},
                                                inputs={
                                                    "yolo_box_input":
                                                    TensorConfig(
                                                        data_gen=partial(
                                                            generate_input1,
                                                            dics, batch,
                                                            class_num)),
                                                    "imgsize": TensorConfig(
                                                        data_gen=partial(
                                                            generate_input2,
                                                            dics, batch))
                                                },
                                                outputs=["boxes", "scores"])

                                            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            if attrs[0]['iou_aware'] == True:
                channel = 3 * (attrs[0]['class_num'] + 6)
                self.dynamic_shape.min_input_shape = {
                    "scale_input": [1, channel, 12, 12]
                }
                self.dynamic_shape.max_input_shape = {
                    "scale_input": [4, channel, 24, 24]
                }
                self.dynamic_shape.opt_input_shape = {
                    "scale_input": [1, channel, 24, 24]
                }
            else:
                channel = 3 * (attrs[0]['class_num'] + 5)
                self.dynamic_shape.min_input_shape = {
                    "scale_input": [1, channel, 12, 12]
                }
                self.dynamic_shape.max_input_shape = {
                    "scale_input": [4, channel, 24, 24]
                }
                self.dynamic_shape.opt_input_shape = {
                    "scale_input": [1, channel, 24, 24]
                }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            if dynamic_shape == True:
                return 0, 5
            else:
                return 1, 4

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]
        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-3

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-3

    def add_skip_trt_case(self):
        pass

    def test(self):
        self.add_skip_trt_case()
        self.run_test()


if __name__ == "__main__":
    unittest.main()
