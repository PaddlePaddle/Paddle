#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from paddle.trainer_config_helpers import *

img = data_layer(name='pixel', size=784)

hidden = fc_layer(
    input=img,
    size=200,
    param_attr=ParamAttr(name='hidden.w'),
    bias_attr=ParamAttr(name='hidden.b'))

prob = fc_layer(
    input=hidden,
    size=10,
    act=SoftmaxActivation(),
    param_attr=ParamAttr(name='prob.w'),
    bias_attr=ParamAttr(name='prob.b'))

outputs(prob)
