# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
import os
import optimizer
import layer
import activation
import parameters
import trainer
import event
import data_type
import topology
import data_feeder
import networks
import evaluator
from . import dataset
from . import reader
from . import plot
import attr
import op
import pooling
import inference
import networks
import py_paddle.swig_paddle as api
import minibatch
import plot
import image

__all__ = [
    'optimizer',
    'layer',
    'activation',
    'parameters',
    'init',
    'trainer',
    'event',
    'data_type',
    'attr',
    'pooling',
    'data_feeder',
    'dataset',
    'reader',
    'topology',
    'networks',
    'infer',
    'plot',
    'evaluator',
    'image',
]


def init(**kwargs):
    args = []
    args_dict = {}
    # NOTE: append arguments if they are in ENV
    for ek, ev in os.environ.iteritems():
        if ek.startswith("PADDLE_INIT_"):
            args_dict[ek.replace("PADDLE_INIT_", "").lower()] = str(ev)

    args_dict.update(kwargs)
    # NOTE: overwrite arguments from ENV if it is in kwargs
    for key in args_dict.keys():
        args.append('--%s=%s' % (key, str(args_dict[key])))

    api.initPaddle(*args)


infer = inference.infer
batch = minibatch.batch
