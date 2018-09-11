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

from paddle.trainer.config_parser import *
__all__ = [
    'HookAttr', 'ParamAttr', 'ExtraAttr', 'ParameterAttribute',
    'ExtraLayerAttribute'
]


def convert_and_compare(x, Type):
    """
    Convert x to be the same type as Type and then convert back to
    check whether there is a loss of information
    :param x: object to be checked
    :param Type: target type to check x over

    """
    return type(x)(Type(x)) == x


def is_compatible_with(x, Type):
    """
    Check if x has a type compatible with Type
    :param x: object to be checked
    :param Type: target type to check x over

    """
    if type(x) == Type:
        return True
    try:
        if float == Type or int == Type:
            # avoid those types that can be converted to float/int but not very
            # meaningful and  could potentially lead to error
            # i.e., str and bool typed value should not be used for initializing float/int variable
            if not isinstance(x, str) and not isinstance(x, bool):
                return convert_and_compare(x, Type)
        elif bool == Type:
            # should not use string type to initialize bool variable
            if not isinstance(x, str):
                return convert_and_compare(x, Type)
        else:
            return False
    except:
        return False


class HookAttribute(object):
    """
    Hook Attribute object. As a member of ParameterAttribute class, the hook is an auxiliary operation that occurs
    during training process of a layer with parameters, such as img_conv layer, fc layer.

    :param  type: Hook type, currently supported types:
                        'pruning' :  user specify a sparsity_ratio before training started, and the
                            network will prune the parameters based on the sparsity_ratio.
                            eg: The definition of Hook object can be hk = HookAttribute('pruning', 0.6)
                            The specific usage can be paddle.layer.img_conv(input=img, filter_size=3,
                                                                       num_channels=3, num_filters=64,
                                                                       param_attr=ParameterAttribute(update_hooks=hk) )
                            The pruning details can be found https://arxiv.org/pdf/1506.02626.pdf
    :type type: string

    :param sparsity_ratio: Must be specified if hook type is 'pruning',
                        it represents the ratio of the zero elements to be set by the Parameter.
    :type sparsity_ratio: float or None

    """

    def __init__(self, type, sparsity_ratio=None):
        self.type = type
        self.sparsity_ratio = sparsity_ratio
        if self.sparsity_ratio is not None:
            assert is_compatible_with(
                self.sparsity_ratio,
                float), 'sparisity_ratio must be float type'
            assert self.sparsity_ratio <= 1 and self.sparsity_ratio >= 0, 'sparsity_ratio must be a float between [0, 1] '

    def __call__(self):
        return ParameterHook(self.type, sparsity_ratio=self.sparsity_ratio)


class ParameterAttribute(object):
    """
    Parameter Attributes object. To fine-tuning network training process, user
    can set attribute to control training details, such as l1,l2 rate / learning
    rate / how to init param.

    NOTE: IT IS A HIGH LEVEL USER INTERFACE.

    :param is_static: True if this parameter will be fixed while training.
    :type is_static: bool

    :param initial_std: Gauss Random initialization standard deviation.
                        None if not using Gauss Random initialize parameter.
    :type initial_std: float or None
    :param initial_mean:  Gauss Random initialization mean.
                         None if not using Gauss Random initialize parameter.
    :type initial_mean: float or None
    :param initial_max: Uniform initialization max value.
    :type initial_max: float or None
    :param initial_min: Uniform initialization min value.
    :type initial_min: float or None
    :param l1_rate: the l1 regularization factor
    :type l1_rate: float or None
    :param l2_rate: the l2 regularization factor
    :type l2_rate: float or None
    :param learning_rate: The parameter learning rate. None means 1.
                          The learning rate when optimize is LEARNING_RATE =
                          GLOBAL_LEARNING_RATE * PARAMETER_LEARNING_RATE
                          * SCHEDULER_FACTOR.

    :type learning_rate: float or None
    :param momentum: The parameter momentum. None means use global value.
    :type momentum: float or None
    :param gradient_clipping_threshold: gradient clipping threshold. If gradient
                                        value larger than some value, will be
                                        clipped.
    :type gradient_clipping_threshold: float
    :param sparse_update: Enable sparse update for this parameter. It will
                          enable both local and remote sparse update.
    :type sparse_update: bool
    :param update_hooks: A HookAttribute object.
    :type update_hooks: HookAttribute
    :param initializer: If not None, it should be a callable object which accepts
                        a parameter name and returns numpy array for the initial
                        value of the parameter
    :type initializer: callable object
    """

    def __init__(self,
                 name=None,
                 is_static=False,
                 initial_std=None,
                 initial_mean=None,
                 initial_max=None,
                 initial_min=None,
                 l1_rate=None,
                 l2_rate=None,
                 learning_rate=None,
                 momentum=None,
                 gradient_clipping_threshold=None,
                 sparse_update=False,
                 update_hooks=None,
                 initializer=None):
        self.attr = {}

        if is_static:
            self.attr['is_static'] = True

        if initial_std is None and initial_mean is None and initial_max \
                is None and initial_min is None:
            self.attr['initial_smart'] = True
        elif is_compatible_with(initial_std, float) or \
             is_compatible_with(initial_mean, float):
            if initial_std is not None:
                self.attr['initial_std'] = initial_std
            if initial_mean is not None:
                self.attr['initial_mean'] = initial_mean
            self.attr['initial_strategy'] = 0  # Gauss Random
        elif is_compatible_with(initial_max, float) and \
             is_compatible_with(initial_min, float):
            initial_max = initial_max
            initial_min = initial_min
            assert initial_min < initial_max
            initial_mean = (initial_max + initial_min) / 2
            initial_std = initial_mean - initial_min
            self.attr['initial_mean'] = initial_mean
            self.attr['initial_std'] = initial_std
            self.attr['initial_strategy'] = 1  # Uniform Random
        else:
            raise RuntimeError("Unexpected branch.")

        if not is_static and is_compatible_with(l1_rate, float):
            self.attr['decay_rate_l1'] = l1_rate

        if not is_static and is_compatible_with(l2_rate, float):
            self.attr['decay_rate'] = l2_rate

        if not is_static and is_compatible_with(learning_rate, float):
            self.attr['learning_rate'] = learning_rate

        if not is_static and is_compatible_with(momentum, float):
            self.attr['momentum'] = momentum

        if name is not None:
            self.attr['parameter_name'] = name

        if sparse_update:
            self.attr['sparse_update'] = True
            self.attr['sparse_remote_update'] = True

        if gradient_clipping_threshold is not None and \
                is_compatible_with(gradient_clipping_threshold, float):
            self.attr['gradient_clipping_threshold'] = \
                gradient_clipping_threshold
        if initializer is not None:
            self.attr['initializer'] = initializer

        if update_hooks:
            self.attr['update_hooks'] = update_hooks

    def set_default_parameter_name(self, name):
        """
        Set default parameter name. If parameter not set, then will use default
        parameter name.


        :param name: default parameter name.
        :type name: basestring
        """
        if 'parameter_name' not in self.attr:
            self.attr['parameter_name'] = name

    @staticmethod
    def to_bias(bias_attr):
        if isinstance(bias_attr, ParameterAttribute):
            return Bias(**bias_attr.attr)
        else:
            return False


class ExtraLayerAttribute(object):
    """
    Some high level layer attributes config. You can set all attributes here,
    but some layer doesn't support all attributes. If you set an attribute to a
    layer that not support this attribute, paddle will print an error and core.

    :param error_clipping_threshold: Error clipping threshold.
    :type error_clipping_threshold: float
    :param drop_rate: Dropout rate. Dropout will create a mask on layer output.
                      The dropout rate is the zero rate of this mask. The
                      details of what dropout is please refer to `JMLRdropout
                      <https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
                      >`_.
    :type drop_rate: float
    :param device: device ID of layer. device=-1, use CPU. device>=0, use GPU.
                   The details allocation in parallel_nn please refer to `use_case
                   <https://github.com/PaddlePaddle/Paddle/blob/develop/doc/v2
                   /howto/cmd_parameter/use_case_en.md#case-2-specify-layers-in
                   -different-devices>`_.
    :type device: int
    """

    def __init__(self,
                 error_clipping_threshold=None,
                 drop_rate=None,
                 device=None):
        self.attr = dict()
        if error_clipping_threshold is not None:
            error_clipping_threshold = float(error_clipping_threshold)
            if error_clipping_threshold < 0:
                raise ValueError("Error clipping must > 0")
            self.attr['error_clipping_threshold'] = error_clipping_threshold
        if drop_rate is not None:
            drop_rate = float(drop_rate)
            if drop_rate < 0:
                raise ValueError("Dropout rate must > 0")
            self.attr["drop_rate"] = drop_rate

        if isinstance(device, int):
            self.attr["device"] = device

    def check(self, layer_name):
        for key in self.attr:
            if not hasattr(self, 'can_%s' % key) or \
                    not getattr(self, 'can_%s' % key):
                raise NotImplementedError("Layer %s does not support %s" %
                                          (layer_name, key))

    @staticmethod
    def to_kwargs(attr):
        if attr is None:
            return dict()
        else:
            return attr.attr


HookAttr = HookAttribute
ParamAttr = ParameterAttribute
ExtraAttr = ExtraLayerAttribute
