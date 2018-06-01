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

from initializer import Initializer, Xavier, Constant
from regularizer import WeightDecayRegularizer

__all__ = [
    'ParamAttr',
    'WeightNormParamAttr',
]


class ParamAttr(object):
    def __init__(self,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 gradient_clip=None,
                 do_model_average=None):
        self.name = name
        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.trainable = trainable
        self.gradient_clip = gradient_clip
        self.model_average = do_model_average

    def set_default_initializer(self, initializer):
        if initializer is None:
            if self.initializer is None:
                raise ValueError("ParamAttr.initializer is not set")
            return

        if self.initializer is not None:
            return

        self.initializer = initializer

    def set_default_param_initializer(self):
        self.set_default_initializer(Xavier())

    def set_default_bias_initializer(self):
        self.set_default_initializer(Constant(0.0))

    @staticmethod
    def to_attr(arg):
        if arg is None:
            return ParamAttr()
        elif isinstance(arg, list) or isinstance(arg, tuple):
            return [ParamAttr.to_attr(a) for a in arg]
        elif isinstance(arg, ParamAttr):
            return arg
        elif isinstance(arg, str) or isinstance(arg, unicode):
            return ParamAttr(name=arg)
        elif isinstance(arg, Initializer):
            return ParamAttr(initializer=arg)
        elif isinstance(arg, WeightDecayRegularizer):
            return ParamAttr(regularizer=arg)
        elif isinstance(arg, bool):
            return ParamAttr.to_attr(None) if arg else False
        else:
            raise TypeError("{0} cast to ParamAttr".format(type(arg)))

    def to_kwargs(self, with_initializer=False):
        kwargs = {
            'name': self.name,
            'optimize_attr': {
                'learning_rate': self.learning_rate
            },
            'regularizer': self.regularizer,
            'trainable': self.trainable,
            'gradient_clip_attr': self.gradient_clip,
            'model_average': self.model_average
        }
        if with_initializer:
            kwargs['initializer'] = self.initializer
        return kwargs


class WeightNormParamAttr(ParamAttr):
    """
    Used for weight normalization. Any field in ParamAttr can also be set here.
    Besides, an extra field dim can be set to indicate the dimension except
    which to normalize.
    """
    # List to record the parameters reparameterized by weight normalization.
    # If these parameters are treated as Variable rather than Parameter,
    # it can be used to discriminate these parameters and help to serialize
    # these paramters for inference.
    params_with_weight_norm = []

    def __init__(self, dim=None, **kwargs):
        super(WeightNormParamAttr, self).__init__(**kwargs)
        self.dim = dim
