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

__all__ = [
    "TanhActivation", "SigmoidActivation", "SoftmaxActivation",
    "IdentityActivation", "LinearActivation", 'SequenceSoftmaxActivation',
    'ExpActivation', "ReluActivation", "BReluActivation", "SoftReluActivation",
    "STanhActivation", "AbsActivation", "SquareActivation", "BaseActivation",
    "LogActivation", "SqrtActivation", "ReciprocalActivation",
    "SoftSignActivation"
]


class BaseActivation(object):
    """
    A mark for activation class.
    Each activation inherit BaseActivation, which has two parameters.

    :param name: activation name in paddle config.
    :type name: basestring
    :param support_hppl: True if supported by hppl. HPPL is a library used by paddle
                         internally. Currently, lstm layer can only use activations
                         supported by hppl.
    :type support_hppl: bool
    """

    def __init__(self, name, support_hppl):
        self.name = name
        self.support_hppl = support_hppl

    def __repr__(self):
        return self.name


class TanhActivation(BaseActivation):
    """
    Tanh activation.

    .. math::

       f(z)=tanh(z)=\\frac{e^z-e^{-z}}{e^z+e^{-z}}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'tanh', True)


class SigmoidActivation(BaseActivation):
    """
    Sigmoid activation.

    .. math::

       f(z) = \\frac{1}{1+exp(-z)}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'sigmoid', True)


class SoftmaxActivation(BaseActivation):
    """
    Softmax activation for simple input



    .. math::

       P(y=j|x) = \\frac{e^{x_j}} {\\sum^K_{k=1} e^{x_k} }
    """

    def __init__(self):
        BaseActivation.__init__(self, 'softmax', False)


class SequenceSoftmaxActivation(BaseActivation):
    """
    Softmax activation for one sequence. The dimension of input feature must be
    1 and a sequence.

    ..  code:: python

        result = softmax(for each_feature_vector[0] in input_feature)
        for i, each_time_step_output in enumerate(output):
            each_time_step_output = result[i]
    """

    def __init__(self):
        BaseActivation.__init__(self, 'sequence_softmax', False)


class IdentityActivation(BaseActivation):
    """
    Identity Activation.

    Just do nothing for output both forward/backward.
    """

    def __init__(self):
        BaseActivation.__init__(self, '', False)


LinearActivation = IdentityActivation


class ReluActivation(BaseActivation):
    """
    Relu activation.

    forward. :math:`y = max(0, z)`

    derivative:

    .. math::

       1  &\\quad if z > 0 \\\\
       0  &\\quad\\mathrm{otherwize}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'relu', True)


class BReluActivation(BaseActivation):
    """
    BRelu Activation.

    forward.  :math:`y = min(24, max(0, z))`

    derivative:

    .. math::

       1  &\\quad if 0 < z < 24 \\\\
       0  &\\quad \\mathrm{otherwise}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'brelu', False)


class SoftReluActivation(BaseActivation):
    """
    SoftRelu Activation.
    """

    def __init__(self):
        BaseActivation.__init__(self, 'softrelu', False)


class STanhActivation(BaseActivation):
    """
    Scaled Tanh Activation.

    .. math::

       f(z) = 1.7159 * tanh(2/3*z)
    """

    def __init__(self):
        BaseActivation.__init__(self, 'stanh', False)


class AbsActivation(BaseActivation):
    """
    Abs Activation.

    Forward:    :math:`f(z) = abs(z)`

    Derivative:

    .. math::

       1 &\\quad if \\quad z > 0 \\\\
       -1 &\\quad if \\quad z < 0 \\\\
       0 &\\quad if \\quad z = 0
    """

    def __init__(self):
        BaseActivation.__init__(self, 'abs', False)


class SquareActivation(BaseActivation):
    """
    Square Activation.

    .. math::
       f(z) = z^2.
    """

    def __init__(self):
        BaseActivation.__init__(self, 'square', False)


class ExpActivation(BaseActivation):
    """
    Exponential Activation.

    .. math::
       f(z) = e^z.
    """

    def __init__(self):
        BaseActivation.__init__(self, 'exponential', False)


class LogActivation(BaseActivation):
    """
    Logarithm Activation.

    .. math::
       f(z) = log(z)
    """

    def __init__(self):
        BaseActivation.__init__(self, 'log', False)


class SqrtActivation(BaseActivation):
    """
    Square Root Activation.

    .. math::
       f(z) = sqrt(z)
    """

    def __init__(self):
        BaseActivation.__init__(self, 'sqrt', False)


class ReciprocalActivation(BaseActivation):
    """
    Reciprocal Activation.

    .. math::
       f(z)=\\frac{1}{z}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'reciprocal', False)


class SoftSignActivation(BaseActivation):
    """
    SoftSign Activation.

    .. math::
       f(z)=\\frac{z}{1 + |z|}
    """

    def __init__(self):
        BaseActivation.__init__(self, 'softsign', False)
