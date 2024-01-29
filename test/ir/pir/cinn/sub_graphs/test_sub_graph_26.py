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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^PeleeNet^PeleeNet
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu||api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[32, 224, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_16 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_17 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_18 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_19 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_20 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_21 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_22 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_23 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_24 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_25 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_26 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_27 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_28 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_29 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_30 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_31 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_32 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_33 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_34 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_35 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_36 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_37 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_38 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_39 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_40 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_41 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_42 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_43 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_44 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_45 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_46 = self.create_parameter(
            shape=[32, 192, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_47 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_48 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_49 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_50 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_51 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_52 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_53 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_54 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.parameter_55 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_56 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_57 = self.create_parameter(
            shape=[64, 288, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_58 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_59 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_60 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_61 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_62 = self.create_parameter(
            shape=[704, 704, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_63 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_64 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_65 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_66 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_67 = self.create_parameter(
            shape=[128, 128, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_68 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_69 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_70 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_71 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_72 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_73 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_74 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_75 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_76 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_77 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_78 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_79 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_80 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_81 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_82 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_83 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_84 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_85 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_86 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_87 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_88 = self.create_parameter(
            shape=[32, 224, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_89 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_90 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_91 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_92 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_93 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_94 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_95 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_96 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_97 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_98 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_99 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_100 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_101 = self.create_parameter(
            shape=[512, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_102 = self.create_parameter(
            shape=[64, 576, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_103 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_104 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_105 = self.create_parameter(
            shape=[64, 416, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_106 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_107 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_108 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_109 = self.create_parameter(
            shape=[32, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_110 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_111 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_112 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_113 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_114 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_115 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_116 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_117 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_118 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_119 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_120 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_121 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_122 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_123 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_124 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_125 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_126 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_127 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_128 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_129 = self.create_parameter(
            shape=[64, 320, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_130 = self.create_parameter(
            shape=[64, 384, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_131 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_132 = self.create_parameter(
            shape=[32, 192, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_133 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_134 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_135 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_136 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_137 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_138 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_139 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_140 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_141 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_142 = self.create_parameter(
            shape=[64, 544, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_143 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_144 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_145 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_146 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_147 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_148 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_149 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_150 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_151 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_152 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_153 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_154 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_155 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_156 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_157 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_158 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_159 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_160 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_161 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_162 = self.create_parameter(
            shape=[32, 160, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_163 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_164 = self.create_parameter(
            shape=[64, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_165 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_166 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_167 = self.create_parameter(
            shape=[64, 448, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_168 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_169 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_170 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_171 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_172 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_173 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_174 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_175 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_176 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_177 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_178 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_179 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_180 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.parameter_181 = self.create_parameter(
            shape=[16, 64, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_182 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_183 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_184 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_185 = self.create_parameter(
            shape=[64, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_186 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_187 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_188 = self.create_parameter(
            shape=[64, 416, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_189 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_190 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_191 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_192 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_193 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_194 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_195 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_196 = self.create_parameter(
            shape=[64, 352, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_197 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_198 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_199 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_200 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_201 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_202 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_203 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_204 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_205 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_206 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_207 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_208 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_209 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_210 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_211 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_212 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_213 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_214 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_215 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_216 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_217 = self.create_parameter(
            shape=[16, 96, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_218 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_219 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_220 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_221 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_222 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_223 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_224 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_225 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_226 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_227 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_228 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_229 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_230 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_231 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_232 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_233 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_234 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_235 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_236 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_237 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_238 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_239 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_240 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_241 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_242 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_243 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_244 = self.create_parameter(
            shape=[64, 384, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_245 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_246 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_247 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_248 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_249 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_250 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_251 = self.create_parameter(
            shape=[704],
            dtype=paddle.float32,
        )
        self.parameter_252 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_253 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_254 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.parameter_255 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_256 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_257 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_258 = self.create_parameter(
            shape=[128],
            dtype=paddle.float32,
        )
        self.parameter_259 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_260 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_261 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_262 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_263 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_264 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_265 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_266 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_267 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_268 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_269 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_270 = self.create_parameter(
            shape=[64, 672, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_271 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_272 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_273 = self.create_parameter(
            shape=[64, 640, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_274 = self.create_parameter(
            shape=[64, 672, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_275 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_276 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_277 = self.create_parameter(
            shape=[64, 288, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_278 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_279 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_280 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_281 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_282 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_283 = self.create_parameter(
            shape=[64, 640, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_284 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_285 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_286 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_287 = self.create_parameter(
            shape=[16, 32, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_288 = self.create_parameter(
            shape=[704],
            dtype=paddle.float32,
        )
        self.parameter_289 = self.create_parameter(
            shape=[32, 128, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_290 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_291 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_292 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_293 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_294 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_295 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_296 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_297 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_298 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_299 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_300 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_301 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_302 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_303 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_304 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_305 = self.create_parameter(
            shape=[64, 448, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_306 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_307 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_308 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_309 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_310 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_311 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_312 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_313 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_314 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_315 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_316 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_317 = self.create_parameter(
            shape=[64, 608, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_318 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_319 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_320 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_321 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_322 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_323 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_324 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_325 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_326 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_327 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_328 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_329 = self.create_parameter(
            shape=[32, 64, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_330 = self.create_parameter(
            shape=[32, 3, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_331 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_332 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_333 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_334 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_335 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_336 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_337 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_338 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_339 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_340 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_341 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_342 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_343 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_344 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_345 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_346 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_347 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_348 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_349 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_350 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_351 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_352 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_353 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_354 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_355 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_356 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_357 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_358 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_359 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_360 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_361 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_362 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_363 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_364 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_365 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_366 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_367 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_368 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_369 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_370 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_371 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_372 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_373 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_374 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_375 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_376 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_377 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_378 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_379 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_380 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_381 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_382 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_383 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_384 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_385 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_386 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_387 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_388 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_389 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_390 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_391 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_392 = self.create_parameter(
            shape=[16, 64, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_393 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_394 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_395 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_396 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_397 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_398 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_399 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_400 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_401 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_402 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_403 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_404 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_405 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_406 = self.create_parameter(
            shape=[64, 480, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_407 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_408 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_409 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_410 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_411 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_412 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_413 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_414 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_415 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_416 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_417 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_418 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_419 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_420 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_421 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_422 = self.create_parameter(
            shape=[704],
            dtype=paddle.float32,
        )
        self.parameter_423 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_424 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_425 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_426 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_427 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_428 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_429 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_430 = self.create_parameter(
            shape=[64, 480, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_431 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_432 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_433 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_434 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_435 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_436 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_437 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_438 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_439 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_440 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_441 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_442 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_443 = self.create_parameter(
            shape=[256, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_444 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_445 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_446 = self.create_parameter(
            shape=[64, 608, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_447 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_448 = self.create_parameter(
            shape=[16, 32, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_449 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_450 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_451 = self.create_parameter(
            shape=[64, 320, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_452 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_453 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_454 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_455 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_456 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_457 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_458 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_459 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_460 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_461 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_462 = self.create_parameter(
            shape=[64, 544, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_463 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_464 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_465 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_466 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_467 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_468 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_469 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_470 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_471 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_472 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_473 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_474 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_475 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_476 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_477 = self.create_parameter(
            shape=[64, 352, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_478 = self.create_parameter(
            shape=[16, 96, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_479 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_480 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_481 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_482 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_483 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_484 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_485 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_486 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_487 = self.create_parameter(
            shape=[64, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_488 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_489 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_490 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_491 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_492 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_493 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_494 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_495 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_496 = self.create_parameter(
            shape=[64, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_497 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_498 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_499 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_500 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_501 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_502 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_503 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_504 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_505 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_506 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_507 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_508 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_509 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_510 = self.create_parameter(
            shape=[16, 32, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_511 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_512 = self.create_parameter(
            shape=[64, 576, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_513 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_514 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_515 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_516 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_517 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_518 = self.create_parameter(
            shape=[32, 128, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_519 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_520 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_521 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_522 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_523 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_524 = self.create_parameter(
            shape=[16, 32, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_525 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_526 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_527 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_528 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_529 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_530 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_531 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_532 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_533 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_534 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_535 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_536 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_537 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_538 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_539 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_540 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_541 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_542 = self.create_parameter(
            shape=[704],
            dtype=paddle.float32,
        )
        self.parameter_543 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_544 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_545 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_546 = self.create_parameter(
            shape=[32, 160, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_547 = self.create_parameter(
            shape=[16, 64, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_548 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_549 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_550 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_551 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_552 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_553 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_554 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_555 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_556 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_557 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_558 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_559 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_560 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_561 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.parameter_562 = self.create_parameter(
            shape=[16, 16, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_563 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.parameter_564 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [86, 3, 224, 224], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_330,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_2 = paddle.nn.functional.norm.batch_norm(
            var_1,
            self.parameter_508,
            self.parameter_73,
            weight=self.parameter_125,
            bias=self.parameter_50,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_3 = paddle.nn.functional.activation.relu(var_2)
        var_4 = paddle.nn.functional.conv._conv_nd(
            var_3,
            self.parameter_287,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_5 = paddle.nn.functional.norm.batch_norm(
            var_4,
            self.parameter_256,
            self.parameter_63,
            weight=self.parameter_182,
            bias=self.parameter_98,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_6 = paddle.nn.functional.activation.relu(var_5)
        var_7 = paddle.nn.functional.conv._conv_nd(
            var_6,
            self.parameter_109,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_8 = paddle.nn.functional.norm.batch_norm(
            var_7,
            self.parameter_249,
            self.parameter_207,
            weight=self.parameter_417,
            bias=self.parameter_349,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_9 = paddle.nn.functional.activation.relu(var_8)
        var_10 = paddle.nn.functional.pooling.max_pool2d(
            var_3,
            kernel_size=2,
            stride=2,
            padding=0,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_11 = paddle.tensor.manipulation.concat([var_10, var_9], 1)
        var_12 = paddle.nn.functional.conv._conv_nd(
            var_11,
            self.parameter_329,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_13 = paddle.nn.functional.norm.batch_norm(
            var_12,
            self.parameter_447,
            self.parameter_359,
            weight=self.parameter_186,
            bias=self.parameter_187,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_14 = paddle.nn.functional.activation.relu(var_13)
        var_15 = paddle.nn.functional.conv._conv_nd(
            var_14,
            self.parameter_448,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_16 = paddle.nn.functional.norm.batch_norm(
            var_15,
            self.parameter_260,
            self.parameter_516,
            weight=self.parameter_214,
            bias=self.parameter_393,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_17 = paddle.nn.functional.activation.relu(var_16)
        var_18 = paddle.nn.functional.conv._conv_nd(
            var_17,
            self.parameter_255,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_19 = paddle.nn.functional.norm.batch_norm(
            var_18,
            self.parameter_230,
            self.parameter_136,
            weight=self.parameter_113,
            bias=self.parameter_402,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_20 = paddle.nn.functional.activation.relu(var_19)
        var_21 = paddle.nn.functional.conv._conv_nd(
            var_14,
            self.parameter_524,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_22 = paddle.nn.functional.norm.batch_norm(
            var_21,
            self.parameter_432,
            self.parameter_160,
            weight=self.parameter_64,
            bias=self.parameter_434,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_23 = paddle.nn.functional.activation.relu(var_22)
        var_24 = paddle.nn.functional.conv._conv_nd(
            var_23,
            self.parameter_41,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_25 = paddle.nn.functional.norm.batch_norm(
            var_24,
            self.parameter_127,
            self.parameter_14,
            weight=self.parameter_525,
            bias=self.parameter_114,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_26 = paddle.nn.functional.activation.relu(var_25)
        var_27 = paddle.nn.functional.conv._conv_nd(
            var_26,
            self.parameter_321,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_28 = paddle.nn.functional.norm.batch_norm(
            var_27,
            self.parameter_16,
            self.parameter_78,
            weight=self.parameter_115,
            bias=self.parameter_450,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_29 = paddle.nn.functional.activation.relu(var_28)
        var_30 = paddle.tensor.manipulation.concat([var_14, var_20, var_29], 1)
        var_31 = paddle.nn.functional.conv._conv_nd(
            var_30,
            self.parameter_392,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_32 = paddle.nn.functional.norm.batch_norm(
            var_31,
            self.parameter_183,
            self.parameter_23,
            weight=self.parameter_331,
            bias=self.parameter_221,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_33 = paddle.nn.functional.activation.relu(var_32)
        var_34 = paddle.nn.functional.conv._conv_nd(
            var_33,
            self.parameter_374,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_35 = paddle.nn.functional.norm.batch_norm(
            var_34,
            self.parameter_509,
            self.parameter_96,
            weight=self.parameter_117,
            bias=self.parameter_555,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_36 = paddle.nn.functional.activation.relu(var_35)
        var_37 = paddle.nn.functional.conv._conv_nd(
            var_30,
            self.parameter_181,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_38 = paddle.nn.functional.norm.batch_norm(
            var_37,
            self.parameter_449,
            self.parameter_157,
            weight=self.parameter_295,
            bias=self.parameter_192,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_39 = paddle.nn.functional.activation.relu(var_38)
        var_40 = paddle.nn.functional.conv._conv_nd(
            var_39,
            self.parameter_320,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_41 = paddle.nn.functional.norm.batch_norm(
            var_40,
            self.parameter_8,
            self.parameter_436,
            weight=self.parameter_353,
            bias=self.parameter_522,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_42 = paddle.nn.functional.activation.relu(var_41)
        var_43 = paddle.nn.functional.conv._conv_nd(
            var_42,
            self.parameter_110,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_44 = paddle.nn.functional.norm.batch_norm(
            var_43,
            self.parameter_335,
            self.parameter_5,
            weight=self.parameter_126,
            bias=self.parameter_475,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_45 = paddle.nn.functional.activation.relu(var_44)
        var_46 = paddle.tensor.manipulation.concat([var_30, var_36, var_45], 1)
        var_47 = paddle.nn.functional.conv._conv_nd(
            var_46,
            self.parameter_478,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_48 = paddle.nn.functional.norm.batch_norm(
            var_47,
            self.parameter_278,
            self.parameter_410,
            weight=self.parameter_563,
            bias=self.parameter_529,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_49 = paddle.nn.functional.activation.relu(var_48)
        var_50 = paddle.nn.functional.conv._conv_nd(
            var_49,
            self.parameter_280,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_51 = paddle.nn.functional.norm.batch_norm(
            var_50,
            self.parameter_153,
            self.parameter_400,
            weight=self.parameter_232,
            bias=self.parameter_262,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_52 = paddle.nn.functional.activation.relu(var_51)
        var_53 = paddle.nn.functional.conv._conv_nd(
            var_46,
            self.parameter_217,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_54 = paddle.nn.functional.norm.batch_norm(
            var_53,
            self.parameter_104,
            self.parameter_533,
            weight=self.parameter_343,
            bias=self.parameter_553,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_55 = paddle.nn.functional.activation.relu(var_54)
        var_56 = paddle.nn.functional.conv._conv_nd(
            var_55,
            self.parameter_562,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_57 = paddle.nn.functional.norm.batch_norm(
            var_56,
            self.parameter_205,
            self.parameter_304,
            weight=self.parameter_225,
            bias=self.parameter_151,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_58 = paddle.nn.functional.activation.relu(var_57)
        var_59 = paddle.nn.functional.conv._conv_nd(
            var_58,
            self.parameter_33,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_60 = paddle.nn.functional.norm.batch_norm(
            var_59,
            self.parameter_513,
            self.parameter_391,
            weight=self.parameter_155,
            bias=self.parameter_243,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_61 = paddle.nn.functional.activation.relu(var_60)
        var_62 = paddle.tensor.manipulation.concat([var_46, var_52, var_61], 1)
        var_63 = paddle.nn.functional.conv._conv_nd(
            var_62,
            self.parameter_67,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_64 = paddle.nn.functional.norm.batch_norm(
            var_63,
            self.parameter_180,
            self.parameter_254,
            weight=self.parameter_54,
            bias=self.parameter_258,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_65 = paddle.nn.functional.activation.relu(var_64)
        var_66 = paddle.nn.functional.pooling.avg_pool2d(
            var_65,
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        var_67 = paddle.nn.functional.conv._conv_nd(
            var_66,
            self.parameter_518,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_68 = paddle.nn.functional.norm.batch_norm(
            var_67,
            self.parameter_301,
            self.parameter_517,
            weight=self.parameter_332,
            bias=self.parameter_106,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_69 = paddle.nn.functional.activation.relu(var_68)
        var_70 = paddle.nn.functional.conv._conv_nd(
            var_69,
            self.parameter_28,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_71 = paddle.nn.functional.norm.batch_norm(
            var_70,
            self.parameter_22,
            self.parameter_350,
            weight=self.parameter_383,
            bias=self.parameter_145,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_72 = paddle.nn.functional.activation.relu(var_71)
        var_73 = paddle.nn.functional.conv._conv_nd(
            var_66,
            self.parameter_289,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_74 = paddle.nn.functional.norm.batch_norm(
            var_73,
            self.parameter_79,
            self.parameter_198,
            weight=self.parameter_191,
            bias=self.parameter_424,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_75 = paddle.nn.functional.activation.relu(var_74)
        var_76 = paddle.nn.functional.conv._conv_nd(
            var_75,
            self.parameter_510,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_77 = paddle.nn.functional.norm.batch_norm(
            var_76,
            self.parameter_380,
            self.parameter_354,
            weight=self.parameter_107,
            bias=self.parameter_74,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_78 = paddle.nn.functional.activation.relu(var_77)
        var_79 = paddle.nn.functional.conv._conv_nd(
            var_78,
            self.parameter_292,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_80 = paddle.nn.functional.norm.batch_norm(
            var_79,
            self.parameter_456,
            self.parameter_363,
            weight=self.parameter_556,
            bias=self.parameter_318,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_81 = paddle.nn.functional.activation.relu(var_80)
        var_82 = paddle.tensor.manipulation.concat([var_66, var_72, var_81], 1)
        var_83 = paddle.nn.functional.conv._conv_nd(
            var_82,
            self.parameter_162,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_84 = paddle.nn.functional.norm.batch_norm(
            var_83,
            self.parameter_366,
            self.parameter_172,
            weight=self.parameter_148,
            bias=self.parameter_360,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_85 = paddle.nn.functional.activation.relu(var_84)
        var_86 = paddle.nn.functional.conv._conv_nd(
            var_85,
            self.parameter_161,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_87 = paddle.nn.functional.norm.batch_norm(
            var_86,
            self.parameter_324,
            self.parameter_285,
            weight=self.parameter_34,
            bias=self.parameter_281,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_88 = paddle.nn.functional.activation.relu(var_87)
        var_89 = paddle.nn.functional.conv._conv_nd(
            var_82,
            self.parameter_546,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_90 = paddle.nn.functional.norm.batch_norm(
            var_89,
            self.parameter_80,
            self.parameter_259,
            weight=self.parameter_266,
            bias=self.parameter_439,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_91 = paddle.nn.functional.activation.relu(var_90)
        var_92 = paddle.nn.functional.conv._conv_nd(
            var_91,
            self.parameter_212,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_93 = paddle.nn.functional.norm.batch_norm(
            var_92,
            self.parameter_195,
            self.parameter_257,
            weight=self.parameter_328,
            bias=self.parameter_435,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_94 = paddle.nn.functional.activation.relu(var_93)
        var_95 = paddle.nn.functional.conv._conv_nd(
            var_94,
            self.parameter_190,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_96 = paddle.nn.functional.norm.batch_norm(
            var_95,
            self.parameter_10,
            self.parameter_226,
            weight=self.parameter_224,
            bias=self.parameter_124,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_97 = paddle.nn.functional.activation.relu(var_96)
        var_98 = paddle.tensor.manipulation.concat([var_82, var_88, var_97], 1)
        var_99 = paddle.nn.functional.conv._conv_nd(
            var_98,
            self.parameter_132,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_100 = paddle.nn.functional.norm.batch_norm(
            var_99,
            self.parameter_455,
            self.parameter_38,
            weight=self.parameter_204,
            bias=self.parameter_2,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_101 = paddle.nn.functional.activation.relu(var_100)
        var_102 = paddle.nn.functional.conv._conv_nd(
            var_101,
            self.parameter_146,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_103 = paddle.nn.functional.norm.batch_norm(
            var_102,
            self.parameter_465,
            self.parameter_219,
            weight=self.parameter_56,
            bias=self.parameter_9,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_104 = paddle.nn.functional.activation.relu(var_103)
        var_105 = paddle.nn.functional.conv._conv_nd(
            var_98,
            self.parameter_46,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_106 = paddle.nn.functional.norm.batch_norm(
            var_105,
            self.parameter_173,
            self.parameter_551,
            weight=self.parameter_534,
            bias=self.parameter_548,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_107 = paddle.nn.functional.activation.relu(var_106)
        var_108 = paddle.nn.functional.conv._conv_nd(
            var_107,
            self.parameter_313,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_109 = paddle.nn.functional.norm.batch_norm(
            var_108,
            self.parameter_72,
            self.parameter_47,
            weight=self.parameter_264,
            bias=self.parameter_271,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_110 = paddle.nn.functional.activation.relu(var_109)
        var_111 = paddle.nn.functional.conv._conv_nd(
            var_110,
            self.parameter_227,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_112 = paddle.nn.functional.norm.batch_norm(
            var_111,
            self.parameter_209,
            self.parameter_169,
            weight=self.parameter_474,
            bias=self.parameter_557,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_113 = paddle.nn.functional.activation.relu(var_112)
        var_114 = paddle.tensor.manipulation.concat(
            [var_98, var_104, var_113], 1
        )
        var_115 = paddle.nn.functional.conv._conv_nd(
            var_114,
            self.parameter_7,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_116 = paddle.nn.functional.norm.batch_norm(
            var_115,
            self.parameter_176,
            self.parameter_163,
            weight=self.parameter_134,
            bias=self.parameter_197,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_117 = paddle.nn.functional.activation.relu(var_116)
        var_118 = paddle.nn.functional.conv._conv_nd(
            var_117,
            self.parameter_468,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_119 = paddle.nn.functional.norm.batch_norm(
            var_118,
            self.parameter_364,
            self.parameter_184,
            weight=self.parameter_296,
            bias=self.parameter_246,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_120 = paddle.nn.functional.activation.relu(var_119)
        var_121 = paddle.nn.functional.conv._conv_nd(
            var_114,
            self.parameter_88,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_122 = paddle.nn.functional.norm.batch_norm(
            var_121,
            self.parameter_372,
            self.parameter_466,
            weight=self.parameter_294,
            bias=self.parameter_540,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_123 = paddle.nn.functional.activation.relu(var_122)
        var_124 = paddle.nn.functional.conv._conv_nd(
            var_123,
            self.parameter_351,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_125 = paddle.nn.functional.norm.batch_norm(
            var_124,
            self.parameter_309,
            self.parameter_521,
            weight=self.parameter_381,
            bias=self.parameter_347,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_126 = paddle.nn.functional.activation.relu(var_125)
        var_127 = paddle.nn.functional.conv._conv_nd(
            var_126,
            self.parameter_29,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_128 = paddle.nn.functional.norm.batch_norm(
            var_127,
            self.parameter_139,
            self.parameter_505,
            weight=self.parameter_135,
            bias=self.parameter_128,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_129 = paddle.nn.functional.activation.relu(var_128)
        var_130 = paddle.tensor.manipulation.concat(
            [var_114, var_120, var_129], 1
        )
        var_131 = paddle.nn.functional.conv._conv_nd(
            var_130,
            self.parameter_443,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_132 = paddle.nn.functional.norm.batch_norm(
            var_131,
            self.parameter_24,
            self.parameter_310,
            weight=self.parameter_558,
            bias=self.parameter_120,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_133 = paddle.nn.functional.activation.relu(var_132)
        var_134 = paddle.nn.functional.pooling.avg_pool2d(
            var_133,
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        var_135 = paddle.nn.functional.conv._conv_nd(
            var_134,
            self.parameter_496,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_136 = paddle.nn.functional.norm.batch_norm(
            var_135,
            self.parameter_365,
            self.parameter_228,
            weight=self.parameter_238,
            bias=self.parameter_85,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_137 = paddle.nn.functional.activation.relu(var_136)
        var_138 = paddle.nn.functional.conv._conv_nd(
            var_137,
            self.parameter_311,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_139 = paddle.nn.functional.norm.batch_norm(
            var_138,
            self.parameter_213,
            self.parameter_340,
            weight=self.parameter_312,
            bias=self.parameter_99,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_140 = paddle.nn.functional.activation.relu(var_139)
        var_141 = paddle.nn.functional.conv._conv_nd(
            var_134,
            self.parameter_185,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_142 = paddle.nn.functional.norm.batch_norm(
            var_141,
            self.parameter_12,
            self.parameter_438,
            weight=self.parameter_404,
            bias=self.parameter_385,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_143 = paddle.nn.functional.activation.relu(var_142)
        var_144 = paddle.nn.functional.conv._conv_nd(
            var_143,
            self.parameter_464,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_145 = paddle.nn.functional.norm.batch_norm(
            var_144,
            self.parameter_290,
            self.parameter_118,
            weight=self.parameter_564,
            bias=self.parameter_282,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_146 = paddle.nn.functional.activation.relu(var_145)
        var_147 = paddle.nn.functional.conv._conv_nd(
            var_146,
            self.parameter_52,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_148 = paddle.nn.functional.norm.batch_norm(
            var_147,
            self.parameter_346,
            self.parameter_116,
            weight=self.parameter_223,
            bias=self.parameter_476,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_149 = paddle.nn.functional.activation.relu(var_148)
        var_150 = paddle.tensor.manipulation.concat(
            [var_134, var_140, var_149], 1
        )
        var_151 = paddle.nn.functional.conv._conv_nd(
            var_150,
            self.parameter_277,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_152 = paddle.nn.functional.norm.batch_norm(
            var_151,
            self.parameter_425,
            self.parameter_319,
            weight=self.parameter_261,
            bias=self.parameter_60,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_153 = paddle.nn.functional.activation.relu(var_152)
        var_154 = paddle.nn.functional.conv._conv_nd(
            var_153,
            self.parameter_210,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_155 = paddle.nn.functional.norm.batch_norm(
            var_154,
            self.parameter_371,
            self.parameter_55,
            weight=self.parameter_140,
            bias=self.parameter_355,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_156 = paddle.nn.functional.activation.relu(var_155)
        var_157 = paddle.nn.functional.conv._conv_nd(
            var_150,
            self.parameter_57,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_158 = paddle.nn.functional.norm.batch_norm(
            var_157,
            self.parameter_345,
            self.parameter_460,
            weight=self.parameter_367,
            bias=self.parameter_276,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_159 = paddle.nn.functional.activation.relu(var_158)
        var_160 = paddle.nn.functional.conv._conv_nd(
            var_159,
            self.parameter_423,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_161 = paddle.nn.functional.norm.batch_norm(
            var_160,
            self.parameter_361,
            self.parameter_550,
            weight=self.parameter_93,
            bias=self.parameter_35,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_162 = paddle.nn.functional.activation.relu(var_161)
        var_163 = paddle.nn.functional.conv._conv_nd(
            var_162,
            self.parameter_499,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_164 = paddle.nn.functional.norm.batch_norm(
            var_163,
            self.parameter_43,
            self.parameter_306,
            weight=self.parameter_544,
            bias=self.parameter_177,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_165 = paddle.nn.functional.activation.relu(var_164)
        var_166 = paddle.tensor.manipulation.concat(
            [var_150, var_156, var_165], 1
        )
        var_167 = paddle.nn.functional.conv._conv_nd(
            var_166,
            self.parameter_129,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_168 = paddle.nn.functional.norm.batch_norm(
            var_167,
            self.parameter_538,
            self.parameter_442,
            weight=self.parameter_211,
            bias=self.parameter_560,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_169 = paddle.nn.functional.activation.relu(var_168)
        var_170 = paddle.nn.functional.conv._conv_nd(
            var_169,
            self.parameter_69,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_171 = paddle.nn.functional.norm.batch_norm(
            var_170,
            self.parameter_87,
            self.parameter_536,
            weight=self.parameter_523,
            bias=self.parameter_286,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_172 = paddle.nn.functional.activation.relu(var_171)
        var_173 = paddle.nn.functional.conv._conv_nd(
            var_166,
            self.parameter_451,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_174 = paddle.nn.functional.norm.batch_norm(
            var_173,
            self.parameter_537,
            self.parameter_373,
            weight=self.parameter_51,
            bias=self.parameter_315,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_175 = paddle.nn.functional.activation.relu(var_174)
        var_176 = paddle.nn.functional.conv._conv_nd(
            var_175,
            self.parameter_413,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_177 = paddle.nn.functional.norm.batch_norm(
            var_176,
            self.parameter_479,
            self.parameter_302,
            weight=self.parameter_137,
            bias=self.parameter_388,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_178 = paddle.nn.functional.activation.relu(var_177)
        var_179 = paddle.nn.functional.conv._conv_nd(
            var_178,
            self.parameter_303,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_180 = paddle.nn.functional.norm.batch_norm(
            var_179,
            self.parameter_291,
            self.parameter_368,
            weight=self.parameter_433,
            bias=self.parameter_491,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_181 = paddle.nn.functional.activation.relu(var_180)
        var_182 = paddle.tensor.manipulation.concat(
            [var_166, var_172, var_181], 1
        )
        var_183 = paddle.nn.functional.conv._conv_nd(
            var_182,
            self.parameter_196,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_184 = paddle.nn.functional.norm.batch_norm(
            var_183,
            self.parameter_30,
            self.parameter_216,
            weight=self.parameter_229,
            bias=self.parameter_492,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_185 = paddle.nn.functional.activation.relu(var_184)
        var_186 = paddle.nn.functional.conv._conv_nd(
            var_185,
            self.parameter_467,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_187 = paddle.nn.functional.norm.batch_norm(
            var_186,
            self.parameter_150,
            self.parameter_240,
            weight=self.parameter_444,
            bias=self.parameter_485,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_188 = paddle.nn.functional.activation.relu(var_187)
        var_189 = paddle.nn.functional.conv._conv_nd(
            var_182,
            self.parameter_477,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_190 = paddle.nn.functional.norm.batch_norm(
            var_189,
            self.parameter_3,
            self.parameter_506,
            weight=self.parameter_389,
            bias=self.parameter_165,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_191 = paddle.nn.functional.activation.relu(var_190)
        var_192 = paddle.nn.functional.conv._conv_nd(
            var_191,
            self.parameter_68,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_193 = paddle.nn.functional.norm.batch_norm(
            var_192,
            self.parameter_539,
            self.parameter_494,
            weight=self.parameter_314,
            bias=self.parameter_334,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_194 = paddle.nn.functional.activation.relu(var_193)
        var_195 = paddle.nn.functional.conv._conv_nd(
            var_194,
            self.parameter_369,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_196 = paddle.nn.functional.norm.batch_norm(
            var_195,
            self.parameter_81,
            self.parameter_375,
            weight=self.parameter_143,
            bias=self.parameter_530,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_197 = paddle.nn.functional.activation.relu(var_196)
        var_198 = paddle.tensor.manipulation.concat(
            [var_182, var_188, var_197], 1
        )
        var_199 = paddle.nn.functional.conv._conv_nd(
            var_198,
            self.parameter_130,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_200 = paddle.nn.functional.norm.batch_norm(
            var_199,
            self.parameter_58,
            self.parameter_20,
            weight=self.parameter_293,
            bias=self.parameter_27,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_201 = paddle.nn.functional.activation.relu(var_200)
        var_202 = paddle.nn.functional.conv._conv_nd(
            var_201,
            self.parameter_218,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_203 = paddle.nn.functional.norm.batch_norm(
            var_202,
            self.parameter_457,
            self.parameter_461,
            weight=self.parameter_333,
            bias=self.parameter_472,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_204 = paddle.nn.functional.activation.relu(var_203)
        var_205 = paddle.nn.functional.conv._conv_nd(
            var_198,
            self.parameter_244,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_206 = paddle.nn.functional.norm.batch_norm(
            var_205,
            self.parameter_394,
            self.parameter_552,
            weight=self.parameter_407,
            bias=self.parameter_237,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_207 = paddle.nn.functional.activation.relu(var_206)
        var_208 = paddle.nn.functional.conv._conv_nd(
            var_207,
            self.parameter_401,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_209 = paddle.nn.functional.norm.batch_norm(
            var_208,
            self.parameter_263,
            self.parameter_108,
            weight=self.parameter_495,
            bias=self.parameter_408,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_210 = paddle.nn.functional.activation.relu(var_209)
        var_211 = paddle.nn.functional.conv._conv_nd(
            var_210,
            self.parameter_1,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_212 = paddle.nn.functional.norm.batch_norm(
            var_211,
            self.parameter_327,
            self.parameter_440,
            weight=self.parameter_507,
            bias=self.parameter_131,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_213 = paddle.nn.functional.activation.relu(var_212)
        var_214 = paddle.tensor.manipulation.concat(
            [var_198, var_204, var_213], 1
        )
        var_215 = paddle.nn.functional.conv._conv_nd(
            var_214,
            self.parameter_105,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_216 = paddle.nn.functional.norm.batch_norm(
            var_215,
            self.parameter_502,
            self.parameter_53,
            weight=self.parameter_222,
            bias=self.parameter_445,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_217 = paddle.nn.functional.activation.relu(var_216)
        var_218 = paddle.nn.functional.conv._conv_nd(
            var_217,
            self.parameter_452,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_219 = paddle.nn.functional.norm.batch_norm(
            var_218,
            self.parameter_36,
            self.parameter_97,
            weight=self.parameter_471,
            bias=self.parameter_339,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_220 = paddle.nn.functional.activation.relu(var_219)
        var_221 = paddle.nn.functional.conv._conv_nd(
            var_214,
            self.parameter_188,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_222 = paddle.nn.functional.norm.batch_norm(
            var_221,
            self.parameter_403,
            self.parameter_194,
            weight=self.parameter_415,
            bias=self.parameter_484,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_223 = paddle.nn.functional.activation.relu(var_222)
        var_224 = paddle.nn.functional.conv._conv_nd(
            var_223,
            self.parameter_15,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_225 = paddle.nn.functional.norm.batch_norm(
            var_224,
            self.parameter_252,
            self.parameter_179,
            weight=self.parameter_206,
            bias=self.parameter_514,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_226 = paddle.nn.functional.activation.relu(var_225)
        var_227 = paddle.nn.functional.conv._conv_nd(
            var_226,
            self.parameter_275,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_228 = paddle.nn.functional.norm.batch_norm(
            var_227,
            self.parameter_357,
            self.parameter_17,
            weight=self.parameter_70,
            bias=self.parameter_199,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_229 = paddle.nn.functional.activation.relu(var_228)
        var_230 = paddle.tensor.manipulation.concat(
            [var_214, var_220, var_229], 1
        )
        var_231 = paddle.nn.functional.conv._conv_nd(
            var_230,
            self.parameter_167,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_232 = paddle.nn.functional.norm.batch_norm(
            var_231,
            self.parameter_486,
            self.parameter_171,
            weight=self.parameter_279,
            bias=self.parameter_405,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_233 = paddle.nn.functional.activation.relu(var_232)
        var_234 = paddle.nn.functional.conv._conv_nd(
            var_233,
            self.parameter_395,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_235 = paddle.nn.functional.norm.batch_norm(
            var_234,
            self.parameter_159,
            self.parameter_532,
            weight=self.parameter_336,
            bias=self.parameter_220,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_236 = paddle.nn.functional.activation.relu(var_235)
        var_237 = paddle.nn.functional.conv._conv_nd(
            var_230,
            self.parameter_305,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_238 = paddle.nn.functional.norm.batch_norm(
            var_237,
            self.parameter_242,
            self.parameter_338,
            weight=self.parameter_248,
            bias=self.parameter_337,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_239 = paddle.nn.functional.activation.relu(var_238)
        var_240 = paddle.nn.functional.conv._conv_nd(
            var_239,
            self.parameter_545,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_241 = paddle.nn.functional.norm.batch_norm(
            var_240,
            self.parameter_431,
            self.parameter_203,
            weight=self.parameter_535,
            bias=self.parameter_121,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_242 = paddle.nn.functional.activation.relu(var_241)
        var_243 = paddle.nn.functional.conv._conv_nd(
            var_242,
            self.parameter_458,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_244 = paddle.nn.functional.norm.batch_norm(
            var_243,
            self.parameter_454,
            self.parameter_480,
            weight=self.parameter_416,
            bias=self.parameter_84,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_245 = paddle.nn.functional.activation.relu(var_244)
        var_246 = paddle.tensor.manipulation.concat(
            [var_230, var_236, var_245], 1
        )
        var_247 = paddle.nn.functional.conv._conv_nd(
            var_246,
            self.parameter_406,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_248 = paddle.nn.functional.norm.batch_norm(
            var_247,
            self.parameter_111,
            self.parameter_100,
            weight=self.parameter_178,
            bias=self.parameter_554,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_249 = paddle.nn.functional.activation.relu(var_248)
        var_250 = paddle.nn.functional.conv._conv_nd(
            var_249,
            self.parameter_265,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_251 = paddle.nn.functional.norm.batch_norm(
            var_250,
            self.parameter_325,
            self.parameter_89,
            weight=self.parameter_298,
            bias=self.parameter_370,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_252 = paddle.nn.functional.activation.relu(var_251)
        var_253 = paddle.nn.functional.conv._conv_nd(
            var_246,
            self.parameter_430,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_254 = paddle.nn.functional.norm.batch_norm(
            var_253,
            self.parameter_92,
            self.parameter_561,
            weight=self.parameter_19,
            bias=self.parameter_215,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_255 = paddle.nn.functional.activation.relu(var_254)
        var_256 = paddle.nn.functional.conv._conv_nd(
            var_255,
            self.parameter_441,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_257 = paddle.nn.functional.norm.batch_norm(
            var_256,
            self.parameter_503,
            self.parameter_379,
            weight=self.parameter_48,
            bias=self.parameter_519,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_258 = paddle.nn.functional.activation.relu(var_257)
        var_259 = paddle.nn.functional.conv._conv_nd(
            var_258,
            self.parameter_61,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_260 = paddle.nn.functional.norm.batch_norm(
            var_259,
            self.parameter_399,
            self.parameter_141,
            weight=self.parameter_233,
            bias=self.parameter_501,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_261 = paddle.nn.functional.activation.relu(var_260)
        var_262 = paddle.tensor.manipulation.concat(
            [var_246, var_252, var_261], 1
        )
        var_263 = paddle.nn.functional.conv._conv_nd(
            var_262,
            self.parameter_101,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_264 = paddle.nn.functional.norm.batch_norm(
            var_263,
            self.parameter_4,
            self.parameter_138,
            weight=self.parameter_316,
            bias=self.parameter_307,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_265 = paddle.nn.functional.activation.relu(var_264)
        var_266 = paddle.nn.functional.pooling.avg_pool2d(
            var_265,
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        var_267 = paddle.nn.functional.conv._conv_nd(
            var_266,
            self.parameter_487,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_268 = paddle.nn.functional.norm.batch_norm(
            var_267,
            self.parameter_419,
            self.parameter_356,
            weight=self.parameter_497,
            bias=self.parameter_489,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_269 = paddle.nn.functional.activation.relu(var_268)
        var_270 = paddle.nn.functional.conv._conv_nd(
            var_269,
            self.parameter_358,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_271 = paddle.nn.functional.norm.batch_norm(
            var_270,
            self.parameter_427,
            self.parameter_342,
            weight=self.parameter_175,
            bias=self.parameter_409,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_272 = paddle.nn.functional.activation.relu(var_271)
        var_273 = paddle.nn.functional.conv._conv_nd(
            var_266,
            self.parameter_164,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_274 = paddle.nn.functional.norm.batch_norm(
            var_273,
            self.parameter_469,
            self.parameter_71,
            weight=self.parameter_144,
            bias=self.parameter_166,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_275 = paddle.nn.functional.activation.relu(var_274)
        var_276 = paddle.nn.functional.conv._conv_nd(
            var_275,
            self.parameter_37,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_277 = paddle.nn.functional.norm.batch_norm(
            var_276,
            self.parameter_362,
            self.parameter_103,
            weight=self.parameter_11,
            bias=self.parameter_411,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_278 = paddle.nn.functional.activation.relu(var_277)
        var_279 = paddle.nn.functional.conv._conv_nd(
            var_278,
            self.parameter_147,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_280 = paddle.nn.functional.norm.batch_norm(
            var_279,
            self.parameter_94,
            self.parameter_149,
            weight=self.parameter_559,
            bias=self.parameter_82,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_281 = paddle.nn.functional.activation.relu(var_280)
        var_282 = paddle.tensor.manipulation.concat(
            [var_266, var_272, var_281], 1
        )
        var_283 = paddle.nn.functional.conv._conv_nd(
            var_282,
            self.parameter_462,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_284 = paddle.nn.functional.norm.batch_norm(
            var_283,
            self.parameter_152,
            self.parameter_376,
            weight=self.parameter_13,
            bias=self.parameter_428,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_285 = paddle.nn.functional.activation.relu(var_284)
        var_286 = paddle.nn.functional.conv._conv_nd(
            var_285,
            self.parameter_31,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_287 = paddle.nn.functional.norm.batch_norm(
            var_286,
            self.parameter_420,
            self.parameter_412,
            weight=self.parameter_470,
            bias=self.parameter_382,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_288 = paddle.nn.functional.activation.relu(var_287)
        var_289 = paddle.nn.functional.conv._conv_nd(
            var_282,
            self.parameter_142,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_290 = paddle.nn.functional.norm.batch_norm(
            var_289,
            self.parameter_387,
            self.parameter_168,
            weight=self.parameter_170,
            bias=self.parameter_473,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_291 = paddle.nn.functional.activation.relu(var_290)
        var_292 = paddle.nn.functional.conv._conv_nd(
            var_291,
            self.parameter_396,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_293 = paddle.nn.functional.norm.batch_norm(
            var_292,
            self.parameter_541,
            self.parameter_250,
            weight=self.parameter_193,
            bias=self.parameter_429,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_294 = paddle.nn.functional.activation.relu(var_293)
        var_295 = paddle.nn.functional.conv._conv_nd(
            var_294,
            self.parameter_323,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_296 = paddle.nn.functional.norm.batch_norm(
            var_295,
            self.parameter_463,
            self.parameter_234,
            weight=self.parameter_42,
            bias=self.parameter_154,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_297 = paddle.nn.functional.activation.relu(var_296)
        var_298 = paddle.tensor.manipulation.concat(
            [var_282, var_288, var_297], 1
        )
        var_299 = paddle.nn.functional.conv._conv_nd(
            var_298,
            self.parameter_102,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_300 = paddle.nn.functional.norm.batch_norm(
            var_299,
            self.parameter_133,
            self.parameter_65,
            weight=self.parameter_297,
            bias=self.parameter_91,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_301 = paddle.nn.functional.activation.relu(var_300)
        var_302 = paddle.nn.functional.conv._conv_nd(
            var_301,
            self.parameter_493,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_303 = paddle.nn.functional.norm.batch_norm(
            var_302,
            self.parameter_549,
            self.parameter_32,
            weight=self.parameter_481,
            bias=self.parameter_90,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_304 = paddle.nn.functional.activation.relu(var_303)
        var_305 = paddle.nn.functional.conv._conv_nd(
            var_298,
            self.parameter_512,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_306 = paddle.nn.functional.norm.batch_norm(
            var_305,
            self.parameter_398,
            self.parameter_75,
            weight=self.parameter_500,
            bias=self.parameter_235,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_307 = paddle.nn.functional.activation.relu(var_306)
        var_308 = paddle.nn.functional.conv._conv_nd(
            var_307,
            self.parameter_547,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_309 = paddle.nn.functional.norm.batch_norm(
            var_308,
            self.parameter_504,
            self.parameter_245,
            weight=self.parameter_390,
            bias=self.parameter_21,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_310 = paddle.nn.functional.activation.relu(var_309)
        var_311 = paddle.nn.functional.conv._conv_nd(
            var_310,
            self.parameter_453,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_312 = paddle.nn.functional.norm.batch_norm(
            var_311,
            self.parameter_284,
            self.parameter_45,
            weight=self.parameter_158,
            bias=self.parameter_208,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_313 = paddle.nn.functional.activation.relu(var_312)
        var_314 = paddle.tensor.manipulation.concat(
            [var_298, var_304, var_313], 1
        )
        var_315 = paddle.nn.functional.conv._conv_nd(
            var_314,
            self.parameter_317,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_316 = paddle.nn.functional.norm.batch_norm(
            var_315,
            self.parameter_200,
            self.parameter_418,
            weight=self.parameter_490,
            bias=self.parameter_18,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_317 = paddle.nn.functional.activation.relu(var_316)
        var_318 = paddle.nn.functional.conv._conv_nd(
            var_317,
            self.parameter_267,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_319 = paddle.nn.functional.norm.batch_norm(
            var_318,
            self.parameter_95,
            self.parameter_326,
            weight=self.parameter_268,
            bias=self.parameter_202,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_320 = paddle.nn.functional.activation.relu(var_319)
        var_321 = paddle.nn.functional.conv._conv_nd(
            var_314,
            self.parameter_446,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_322 = paddle.nn.functional.norm.batch_norm(
            var_321,
            self.parameter_322,
            self.parameter_526,
            weight=self.parameter_40,
            bias=self.parameter_414,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_323 = paddle.nn.functional.activation.relu(var_322)
        var_324 = paddle.nn.functional.conv._conv_nd(
            var_323,
            self.parameter_421,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_325 = paddle.nn.functional.norm.batch_norm(
            var_324,
            self.parameter_459,
            self.parameter_174,
            weight=self.parameter_384,
            bias=self.parameter_59,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_326 = paddle.nn.functional.activation.relu(var_325)
        var_327 = paddle.nn.functional.conv._conv_nd(
            var_326,
            self.parameter_272,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_328 = paddle.nn.functional.norm.batch_norm(
            var_327,
            self.parameter_231,
            self.parameter_44,
            weight=self.parameter_39,
            bias=self.parameter_299,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_329 = paddle.nn.functional.activation.relu(var_328)
        var_330 = paddle.tensor.manipulation.concat(
            [var_314, var_320, var_329], 1
        )
        var_331 = paddle.nn.functional.conv._conv_nd(
            var_330,
            self.parameter_273,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_332 = paddle.nn.functional.norm.batch_norm(
            var_331,
            self.parameter_236,
            self.parameter_511,
            weight=self.parameter_189,
            bias=self.parameter_122,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_333 = paddle.nn.functional.activation.relu(var_332)
        var_334 = paddle.nn.functional.conv._conv_nd(
            var_333,
            self.parameter_123,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_335 = paddle.nn.functional.norm.batch_norm(
            var_334,
            self.parameter_437,
            self.parameter_269,
            weight=self.parameter_201,
            bias=self.parameter_344,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_336 = paddle.nn.functional.activation.relu(var_335)
        var_337 = paddle.nn.functional.conv._conv_nd(
            var_330,
            self.parameter_283,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_338 = paddle.nn.functional.norm.batch_norm(
            var_337,
            self.parameter_300,
            self.parameter_239,
            weight=self.parameter_83,
            bias=self.parameter_397,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_339 = paddle.nn.functional.activation.relu(var_338)
        var_340 = paddle.nn.functional.conv._conv_nd(
            var_339,
            self.parameter_520,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_341 = paddle.nn.functional.norm.batch_norm(
            var_340,
            self.parameter_86,
            self.parameter_488,
            weight=self.parameter_0,
            bias=self.parameter_377,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_342 = paddle.nn.functional.activation.relu(var_341)
        var_343 = paddle.nn.functional.conv._conv_nd(
            var_342,
            self.parameter_426,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_344 = paddle.nn.functional.norm.batch_norm(
            var_343,
            self.parameter_112,
            self.parameter_352,
            weight=self.parameter_386,
            bias=self.parameter_156,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_345 = paddle.nn.functional.activation.relu(var_344)
        var_346 = paddle.tensor.manipulation.concat(
            [var_330, var_336, var_345], 1
        )
        var_347 = paddle.nn.functional.conv._conv_nd(
            var_346,
            self.parameter_274,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_348 = paddle.nn.functional.norm.batch_norm(
            var_347,
            self.parameter_241,
            self.parameter_247,
            weight=self.parameter_119,
            bias=self.parameter_498,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_349 = paddle.nn.functional.activation.relu(var_348)
        var_350 = paddle.nn.functional.conv._conv_nd(
            var_349,
            self.parameter_531,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_351 = paddle.nn.functional.norm.batch_norm(
            var_350,
            self.parameter_308,
            self.parameter_348,
            weight=self.parameter_482,
            bias=self.parameter_378,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_352 = paddle.nn.functional.activation.relu(var_351)
        var_353 = paddle.nn.functional.conv._conv_nd(
            var_346,
            self.parameter_270,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_354 = paddle.nn.functional.norm.batch_norm(
            var_353,
            self.parameter_6,
            self.parameter_76,
            weight=self.parameter_483,
            bias=self.parameter_253,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_355 = paddle.nn.functional.activation.relu(var_354)
        var_356 = paddle.nn.functional.conv._conv_nd(
            var_355,
            self.parameter_515,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_357 = paddle.nn.functional.norm.batch_norm(
            var_356,
            self.parameter_527,
            self.parameter_341,
            weight=self.parameter_25,
            bias=self.parameter_49,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_358 = paddle.nn.functional.activation.relu(var_357)
        var_359 = paddle.nn.functional.conv._conv_nd(
            var_358,
            self.parameter_528,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_360 = paddle.nn.functional.norm.batch_norm(
            var_359,
            self.parameter_26,
            self.parameter_66,
            weight=self.parameter_77,
            bias=self.parameter_543,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_361 = paddle.nn.functional.activation.relu(var_360)
        var_362 = paddle.tensor.manipulation.concat(
            [var_346, var_352, var_361], 1
        )
        var_363 = paddle.nn.functional.conv._conv_nd(
            var_362,
            self.parameter_62,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_364 = paddle.nn.functional.norm.batch_norm(
            var_363,
            self.parameter_422,
            self.parameter_288,
            weight=self.parameter_542,
            bias=self.parameter_251,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_365 = paddle.nn.functional.activation.relu(var_364)
        return var_365


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[86, 3, 224, 224], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    # NOTE prim + cinn lead to error
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
