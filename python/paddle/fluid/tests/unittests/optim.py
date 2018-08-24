# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers


class LearningRateScheduler(object):
    """
    Wrapper for learning rate scheduling as described in the Transformer paper.
    LearningRateScheduler adapts the learning rate externally and the adapted
    learning rate will be feeded into the main_program as input data.
    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 learning_rate=0.001,
                 current_steps=0,
                 name="learning_rate"):
        self.current_steps = current_steps
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.static_lr = learning_rate
        self.learning_rate = layers.create_global_var(
            name=name,
            shape=[1],
            value=float(learning_rate),
            dtype="float32",
            persistable=True)

    def update_learning_rate(self):
        self.current_steps += 1
        lr_value = np.power(self.d_model, -0.5) * np.min([
            np.power(self.current_steps, -0.5),
            np.power(self.warmup_steps, -1.5) * self.current_steps
        ]) * self.static_lr
        return np.array([lr_value], dtype="float32")
