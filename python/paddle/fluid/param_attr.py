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

from __future__ import print_function

import six

from .initializer import Initializer, Xavier, Constant
from .regularizer import WeightDecayRegularizer

__all__ = [
    'ParamAttr',
    'WeightNormParamAttr',
]


class ParamAttr(object):
    """
    Create a object to represent the attribute of parameter. The attributes are:
    name, initializer, learning rate, regularizer, trainable, gradient clip,
    and model average.

    Parameters:
        name (str, optional): The parameter's name. Default None, meaning that the name
                would be created automatically.
        initializer (Initializer, optional): The method to initial this parameter. Default
                None, meaning that the weight parameter is initialized by Xavier initializer,
                and the bias parameter is initialized by 0.
        learning_rate (float): The parameter's learning rate. The learning rate when
                optimize is the global learning rates times the parameter's learning rate times
                the factor of learning rate scheduler. Default 1.0.
        regularizer (WeightDecayRegularizer, optional): Regularization factor. Default None, meaning
                there is no regularization.
        trainable (bool): Whether this parameter is trainable. Default True.
        gradient_clip (BaseGradientClipAttr, optional): The method to clip this parameter's
                gradient. Default None, meaning that there is no gradient clip.
        do_model_average (bool): Whether this parameter should do model average
                when model average is enabled. Default False.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                            learning_rate=0.5,
                                            regularizer=fluid.regularizer.L2Decay(1.0),
                                            trainable=True)
            print(w_param_attrs.name) # "fc_weight"
            x = fluid.data(name='X', shape=[None, 1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=10, param_attr=w_param_attrs)
    """

    def __init__(self,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 gradient_clip=None,
                 do_model_average=True):
        self.name = name
        if isinstance(self.name, six.string_types) and self.name == "":
            raise ValueError("name of ParamAttr can not be empty str")

        self.initializer = initializer
        self.learning_rate = learning_rate
        self.regularizer = regularizer
        self.trainable = trainable
        self.gradient_clip = gradient_clip
        self.do_model_average = do_model_average

    def _set_default_initializer(self, initializer):
        """
        Set the default initializer, the initializer should be Constant,
        Uniform, Normal, Xavier, MSRA.

        Args:
            initializer(Initializer): the initializer to set.

        Returns:
            None
        """
        if initializer is None:
            if self.initializer is None:
                raise ValueError("ParamAttr.initializer is not set")
            return

        if self.initializer is not None:
            return

        self.initializer = initializer

    def _set_default_param_initializer(self):
        """
        Set the default initializer for the parameter with Xavier.

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Xavier())

    def _set_default_bias_initializer(self):
        """
        Set the default initializer for the bias with Constant(0.0).

        Args:
            None.

        Returns:
            None.
        """
        self._set_default_initializer(Constant(0.0))

    @staticmethod
    def _to_attr(arg):
        """
        Create ParamAttr[s].

        Args:
            arg: Arguments to initialize ParamAttr[s]. arg's type can be
                str, Initializer, float, WeightDecayRegularizer, BaseGradientClipAttr,
                bool, ParamAttr, or a list of above type.

        Returns:
            ParamAttr[s]: ParamAttr[s] initialized with arg.

        Raises:
            arg can not initialize a ParamAttr.
        """
        if arg is None:
            return ParamAttr()
        elif isinstance(arg, list) or isinstance(arg, tuple):
            return [ParamAttr._to_attr(a) for a in arg]
        elif isinstance(arg, ParamAttr):
            return arg
        elif isinstance(arg, six.string_types):
            return ParamAttr(name=arg)
        elif isinstance(arg, Initializer):
            return ParamAttr(initializer=arg)
        elif isinstance(arg, WeightDecayRegularizer):
            return ParamAttr(regularizer=arg)
        elif isinstance(arg, bool):
            return ParamAttr._to_attr(None) if arg else False
        else:
            raise TypeError("{0} cast to ParamAttr".format(type(arg)))

    def _to_kwargs(self, with_initializer=False):
        """
        Returns the attributes of this parameter.

        Args:
            with_initializer(bool): Whether to add initializer attr.

        Returns:
            Parameter attributes(map): The attributes of this parameter.
        """
        kwargs = {
            'name': self.name,
            'optimize_attr': {
                'learning_rate': self.learning_rate
            },
            'regularizer': self.regularizer,
            'trainable': self.trainable,
            'gradient_clip_attr': self.gradient_clip,
            'do_model_average': self.do_model_average
        }
        if with_initializer:
            kwargs['initializer'] = self.initializer
        return kwargs


class WeightNormParamAttr(ParamAttr):
    """
    Parameter of weight Norm. Weight Norm is a reparameterization of the weight vectors
    in a neural network that decouples the magnitude of those weight vectors from
    their direction. Weight Norm has been implemented as discussed in this
    paper: `Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks
    <https://arxiv.org/pdf/1602.07868.pdf>`_.

    Args:
        dim(int): Dimension over which to compute the norm. Dim is a non-negative
            number which is less than the rank of weight Tensor. For Example, dim can
            be chosen from 0, 1, 2, 3 for convolution whose weight shape is [cout, cin, kh, kw]
            and rank is 4. Default None, meaning that all elements will be normalized.
        name(str, optional): The parameter's name. Default None, meaning that the name would
            be created automatically. Please refer to :ref:`api_guide_Name` for more details.
        initializer(Initializer): The method to initialize this parameter, such as
            ``initializer = fluid.initializer.ConstantInitializer(1.0)``. Default None,
            meaning that the weight parameter is initialized by Xavier initializer, and
            the bias parameter is initialized by 0.
        learning_rate(float32): The parameter's learning rate when
            optimizer is :math:`global\_lr * parameter\_lr * scheduler\_factor`.
            Default 1.0.
        regularizer(WeightDecayRegularizer): Regularization factor, such as
            ``regularizer = fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.1)``.
            Default None, meaning that there is no regularization.
        trainable(bool, optional): Whether this parameter is trainable. Default True.
        gradient_clip: The method to clip this parameter's gradient, such as
            ``gradient_clip = fluid.clip.GradientClipByNorm(clip_norm=2.0))`` .
            Default None, meaning that there is no gradient clip.
        do_model_average(bool, optional): Whether this parameter should do model average.
            Default False.

    Examples:
        .. code-block:: python
            
            import paddle.fluid as fluid
            data = fluid.layers.data(name="data", shape=[3, 32, 32], dtype="float32")
            fc = fluid.layers.fc(input=data,
                                 size=1000,
                                 param_attr=fluid.WeightNormParamAttr(
                                          dim=None,
                                          name='weight_norm_param',
                                          initializer=fluid.initializer.ConstantInitializer(1.0),
                                          learning_rate=1.0,
                                          regularizer=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.1),
                                          trainable=True,
                                          gradient_clip=fluid.clip.GradientClipByNorm(clip_norm=2.0),
                                          do_model_average=False))

    """
    # List to record the parameters reparameterized by weight normalization.
    # If these parameters are treated as Variable rather than Parameter,
    # it can be used to discriminate these parameters and help to serialize
    # these paramters for inference.
    params_with_weight_norm = []

    def __init__(self,
                 dim=None,
                 name=None,
                 initializer=None,
                 learning_rate=1.0,
                 regularizer=None,
                 trainable=True,
                 gradient_clip=None,
                 do_model_average=False):
        super(WeightNormParamAttr, self).__init__(
            name=name,
            initializer=initializer,
            learning_rate=learning_rate,
            regularizer=regularizer,
            trainable=trainable,
            gradient_clip=gradient_clip,
            do_model_average=do_model_average)
        self.dim = dim
