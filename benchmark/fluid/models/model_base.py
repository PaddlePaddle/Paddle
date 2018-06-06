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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse

import paddle.fluid as fluid
from paddle.fluid.regularizer import L1DecayRegularizer
from paddle.fluid.regularizer import L2DecayRegularizer
from paddle.fluid.clip import GradientClipByNorm
from paddle.fluid.clip import GradientClipByGlobalNorm
from paddle.fluid.clip import ErrorClipByValue

__all__ = [
    'get_decay_learning_rate',
    'get_regularization',
    'set_error_clip',
    'set_gradient_clip',
]


def get_decay_learning_rate(decay_method,
                            learning_rate=0.001,
                            decay_steps=100000,
                            decay_rate=0.5,
                            staircase=True):
    if not decay_method:
        return learning_rate
    else:
        decay_op = getattr(fluid.layers, "%s_decay" % decay_method)
        return decay_op(
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)


def get_regularization(regularizer_method, regularizer_coeff=0.1):
    if not regularizer_method:
        return None
    else:
        RegularizerClazz = globals()["%sDecayRegularizer" % regularizer_method]
        regularizer = RegularizerClazz(regularization_coeff=regularizer_coeff)
        return regularizer


def set_error_clip(clip_method,
                   layer_name,
                   clip_min=-1e-6,
                   clip_max=2e-6,
                   program=None):
    assert clip_min < clip_max
    if not clip_method:
        return None
    else:
        ClipClazz = globals()["ErrorClipBy%s" % clip_method]
        if not program:
            prog = fluid.default_main_program()
        else:
            prog = program
        prog.block(0).var(layer_name).set_error_clip(
            ClipClazz(
                max=clip_max, min=clip_min))


def set_gradient_clip(clip_method, clip_norm=1.):
    if not clip_method:
        return None
    else:
        ClipClazz = globals()["GradientClipBy%s" % clip_method]
        fluid.clip.set_gradient_clip(ClipClazz(clip_norm=clip_norm))
        return clip_method
