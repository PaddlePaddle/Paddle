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

from . import layers
from . import framework

__all__ = ['WeightDecay', ]


class WeightDecayBase(object):
    def __str__(self):
        raise NotImplementedError()

    def _append_weight_decay_op(self, param):
        raise NotImplementedError()


class WeightDecay(WeightDecayBase):
    """
    WeightDecay is used to decay the weight.

    Given a tensor t, this operation clips its value to min and max inplace.

    - Any values less than min are set to min.
    - Any values greater than max are set to max.

    Args:
        max (float): The maximum value to clip by.
        min (float, optional): The minimum value to clip by. if not set by user, \
        will be set to -max by framework.

    Examples:
        .. code-block:: python

            var = fluid.framework.Variable(..., error_clip=ErrorClipByValue(max=5.0), ...)
    """

    def __init__(self,
                 main_program=None,
                 attempt_decay_param_fun=None,
                 coeff=0.0):

        if main_program is None:
            main_program = fluid.default_main_program()

        self.param_list_ = main_program.block(0).all_parameters()
        self.scaled_params_ = dict()
        self.params_name_ = []
        self.attempt_decay_param_fun_ = attempt_decay_param_fun

        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Variable.")

        for p in self.param_list_:
            assert isinstance(p, framework.Parameter)
            if self.attempt_decay_param_fun_ is not None and not self.attempt_decay_param_fun_(
                    p.name):
                continue

            with framework.name_scope('weight_decay'):
                assert p.name not in self.params_name_
                self.scaled_params_[p.name] = (p, p * coeff)
                self.params_name_.append(p.name)

    def decay(self):
        """Add weight decay ops to network

        """
        for p_name in self.scaled_params_.keys():
            with framework.name_scope('weight_decay'):
                layers.elementwise_sub(
                    x=self.scaled_params_[p_name][0],
                    y=self.scaled_params_[p_name][1],
                    out=self.scaled_params_[p_name][0])

    def __str__(self):
        info = "Weight Decay, params: "
        for p in self.params_name_:
            info += p
            info += ", "

        return info


WeightDecay = WeightDecay
