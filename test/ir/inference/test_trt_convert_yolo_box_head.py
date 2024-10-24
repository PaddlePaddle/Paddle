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

from __future__ import annotations

import unittest
from functools import partial
from typing import Any

import numpy as np
from program_config import ProgramConfig, TensorConfig
from trt_layer_auto_scan_test import TrtLayerAutoScanTest

import paddle.inference as paddle_infer


class TrtConvertYoloBoxHeadTest(TrtLayerAutoScanTest):
    def sample_program_configs(self):
        def generate_input(attrs: list[dict[str, Any]], batch, shape):
            gen_shape = shape.copy()
            gen_shape.insert(0, batch)
            return np.random.uniform(0, 1, gen_shape).astype("float32")

        input_shape = [[255, 19, 19], [255, 38, 38], [255, 76, 76]]
        anchors = [
            [116, 90, 156, 198, 373, 326],
            [30, 61, 62, 45, 59, 119],
            [10, 13, 16, 30, 33, 23],
        ]
        class_num = 80
        for batch in [1, 4]:
            for i in range(len(anchors)):
                attrs_dict = {
                    "anchors": anchors[i],
                    "class_num": class_num,
                }
                ops_config = [
                    {
                        "op_type": "yolo_box_head",
                        "op_inputs": {
                            "X": ["yolo_box_head_input"],
                        },
                        "op_outputs": {
                            "Out": ["yolo_box_head_output"],
                        },
                        "op_attrs": attrs_dict,
                    }
                ]
                ops = self.generate_op_config(ops_config)
                program_config = ProgramConfig(
                    ops=ops,
                    weights={},
                    inputs={
                        "yolo_box_head_input": TensorConfig(
                            data_gen=partial(
                                generate_input,
                                attrs_dict,
                                batch,
                                input_shape[i],
                            )
                        )
                    },
                    outputs=["yolo_box_head_output"],
                )

                yield program_config

    def sample_predictor_configs(
        self, program_config
    ) -> tuple[paddle_infer.Config, list[int], float]:
        # for static_shape
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        program_config.set_input_type(np.float32)
        yield self.create_inference_config(), [0, 3], 1e-5

    def test(self):
        self.run_test()


if __name__ == "__main__":
    unittest.main()
