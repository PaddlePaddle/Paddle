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

from paddle.trainer.config_parser import default_decay_rate, \
    default_gradient_clipping_threshold, default_momentum

__all__ = [
    'OptimizerType', 'BaseSGDOptimizer', 'MomentumOptimizer', 'AdamaxOptimizer',
    'AdamOptimizer', 'AdaGradOptimizer', 'RMSPropOptimizer',
    'DecayedAdaGradOptimizer', 'AdaDeltaOptimizer', 'BaseRegularization',
    'L2Regularization', 'settings', 'ModelAverage'
]


class OptimizerType(object):
    def to_setting_kwargs(self):
        raise NotImplementedError()

    def extra_settings(self):
        pass

    @property
    def is_support_sparse(self):
        return True


class BaseSGDOptimizer(OptimizerType):
    """
    SGD Optimizer.

    SGD is an optimization method, trying to find a neural network that
    minimize the "cost/error" of it by iteration. In paddle's implementation
    SGD Optimizer is synchronized, which means all gradients will be wait to
    calculate and reduced into one gradient, then do optimize operation.

    The neural network consider the learning problem of minimizing an objective
    function, that has the form of a sum

    ..  math::

        Q(w) = \\sum_{i}^{n} Q_i(w)

    The value of function Q sometimes is the cost of neural network (Mean
    Square Error between prediction and label for example). The function Q is
    parametrised by w, the weight/bias of neural network. And weights is what to
    be learned. The i is the i-th observation in (trainning) data.

    So, the SGD method will optimize the weight by

    ..  math::

        w = w - \\eta \\nabla Q(w) = w - \\eta \\sum_{i}^{n} \\nabla Q_i(w)

    where :math:`\\eta` is learning rate. And :math:`n` is batch size.
    """

    def to_setting_kwargs(self):
        raise NotImplementedError()


class MomentumOptimizer(BaseSGDOptimizer):
    """
    MomentumOptimizer.

    When sparse=True, the update scheme:

    ..  math::

        \\alpha_t &= \\alpha_{t-1} / k \\\\
        \\beta_t &= \\beta_{t-1} / (1 + \\lambda \\gamma_t) \\\\
        u_t &= u_{t-1} - \\alpha_t \\gamma_t g_t \\\\
        v_t &= v_{t-1} + \\tau_{t-1} \\alpha_t \\gamma_t g_t \\\\
        \\tau_t &= \\tau_{t-1} + \\beta_t / \\alpha_t
    
    where :math:`k` is momentum, :math:`\\lambda` is decay rate, 
    :math:`\\gamma_t` is learning rate at the t'th step.

    :param sparse: with sparse support or not.
    :type sparse: bool
    """

    def extra_settings(self):
        default_momentum(self.momentum)

    def to_setting_kwargs(self):
        if self.sparse:
            return {'learning_method': 'sparse_momentum'}
        else:
            return {'learning_method': 'momentum'}

    def __init__(self, momentum=None, sparse=False):
        self.momentum = momentum
        self.sparse = sparse


class AdamOptimizer(BaseSGDOptimizer):
    """
    Adam optimizer.
    The details of please refer `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_

    ..  math::

        m(w, t) & = \\beta_1 m(w, t-1) + (1 - \\beta_1) \\nabla Q_i(w) \\\\
        v(w, t) & = \\beta_2 v(w, t-1) + (1 - \\beta_2)(\\nabla Q_i(w)) ^2 \\\\
        w & = w - \\frac{\\eta}{\\sqrt{v(w,t) + \\epsilon}}

    :param beta1: the :math:`\\beta_1` in equation.
    :type beta1: float
    :param beta2: the :math:`\\beta_2` in equation.
    :type beta2: float
    :param epsilon: the :math:`\\epsilon` in equation. It is used to prevent
                        divided by zero.
    :type epsilon: float
    """

    @property
    def is_support_sparse(self):
        return False

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def to_setting_kwargs(self):
        return {
            'learning_method': 'adam',
            'adam_beta1': self.beta1,
            'adam_beta2': self.beta2,
            'adam_epsilon': self.epsilon
        }


class AdamaxOptimizer(BaseSGDOptimizer):
    """
    Adamax optimizer.

    The details of please refer this `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_

    ..  math::

        m_t & = \\beta_1 * m_{t-1} + (1-\\beta_1)* \\nabla Q_i(w) \\\\
        u_t & = max(\\beta_2*u_{t-1}, abs(\\nabla Q_i(w))) \\\\
        w_t & = w_{t-1} - (\\eta/(1-\\beta_1^t))*m_t/u_t

    :param beta1: the :math:`\\beta_1` in the equation.
    :type beta1: float
    :param beta2: the :math:`\\beta_2` in the equation.
    :type beta2: float
    """

    def __init__(self, beta1, beta2):
        self.beta1 = beta1
        self.beta2 = beta2

    def to_setting_kwargs(self):
        return {
            'learning_method': 'adamax',
            'adam_beta1': self.beta1,
            'adam_beta2': self.beta2
        }

    @property
    def is_support_sparse(self):
        return False


class AdaGradOptimizer(BaseSGDOptimizer):
    """
    Adagrad(for ADAptive GRAdient algorithm) optimizer.

    For details please refer this `Adaptive Subgradient Methods for
    Online Learning and Stochastic Optimization
    <http://www.magicbroom.info/Papers/DuchiHaSi10.pdf>`_.

    ..  math::

        G &= \\sum_{\\tau=1}^{t} g_{\\tau} g_{\\tau}^T \\\\
        w & = w - \\eta diag(G)^{-\\frac{1}{2}} \\circ g
    """

    def to_setting_kwargs(self):
        return {'learning_method': 'adagrad'}

    def __init__(self):
        pass


class RMSPropOptimizer(BaseSGDOptimizer):
    """
    RMSProp(for Root Mean Square Propagation) optimizer. For details please
    refer this `slide <http://www.cs.toronto.edu/~tijmen/csc321/slides/
    lecture_slides_lec6.pdf>`_.

    The equations of this method as follows:

    ..  math::

        v(w, t) & = \\rho v(w, t-1) + (1 - \\rho)(\\nabla Q_{i}(w))^2 \\\\
        w & = w - \\frac{\\eta} {\\sqrt{v(w,t) + \\epsilon}} \\nabla Q_{i}(w)

    :param rho: the :math:`\\rho` in the equation. The forgetting factor.
    :type rho: float
    :param epsilon: the :math:`\\epsilon` in the equation.
    :type epsilon: float
    """

    def to_setting_kwargs(self):
        return {
            'learning_method': 'rmsprop',
            'ada_rou': self.rho,
            'ada_epsilon': self.epsilon
        }

    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon


class DecayedAdaGradOptimizer(BaseSGDOptimizer):
    """
    AdaGrad method with decayed sum gradients. The equations of this method
    show as follow.

    ..  math::

        E(g_t^2) &= \\rho * E(g_{t-1}^2) + (1-\\rho) * g^2 \\\\
        learning\\_rate &= 1/sqrt( ( E(g_t^2) + \\epsilon )

    :param rho: The :math:`\\rho` parameter in that equation
    :type rho: float
    :param epsilon: The :math:`\\epsilon` parameter in that equation.
    :type epsilon: float
    """

    def to_setting_kwargs(self):
        return {
            'learning_method': 'decayed_adagrad',
            'ada_rou': self.rho,
            'ada_epsilon': self.epsilon
        }

    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon


class AdaDeltaOptimizer(BaseSGDOptimizer):
    """
    AdaDelta method. The details of adadelta please refer to this
    `ADADELTA: AN ADAPTIVE LEARNING RATE METHOD
    <http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf>`_.

    ..  math::

        E(g_t^2) &= \\rho * E(g_{t-1}^2) + (1-\\rho) * g^2 \\\\
        learning\\_rate &= sqrt( ( E(dx_{t-1}^2) + \\epsilon ) / ( \\
                          E(g_t^2) + \\epsilon ) ) \\\\
        E(dx_t^2) &= \\rho * E(dx_{t-1}^2) + (1-\\rho) * (-g*learning\\_rate)^2

    :param rho: :math:`\\rho` in equation
    :type rho: float
    :param epsilon: :math:`\\rho` in equation
    :type epsilon: float
    """

    def to_setting_kwargs(self):
        return {
            'learning_method': 'adadelta',
            'ada_rou': self.rho,
            'ada_epsilon': self.epsilon
        }

    def __init__(self, rho=0.95, epsilon=1e-6):
        self.rho = rho
        self.epsilon = epsilon


class BaseRegularization(OptimizerType):
    def __init__(self):
        self.algorithm = ""
        self.learning_method = ""

    def to_setting_kwargs(self):
        return {}


class L2Regularization(BaseRegularization):
    def __init__(self, rate):
        super(L2Regularization, self).__init__()
        self.decay_rate = rate

    def to_setting_kwargs(self):
        if self.algorithm == 'owlqn':
            return {'l2weight': self.decay_rate}
        else:
            return dict()

    def extra_settings(self):
        if self.algorithm == 'sgd' or self.algorithm == 'async_sgd':
            default_decay_rate(self.decay_rate)


class ModelAverage(OptimizerType):
    def to_setting_kwargs(self):
        return {
            'average_window': self.average_window,
            'max_average_window': self.max_average_window,
            'do_average_in_cpu': self.do_average_in_cpu
        }

    def __init__(self,
                 average_window,
                 max_average_window=None,
                 do_average_in_cpu=False):
        self.average_window = average_window
        self.max_average_window = max_average_window
        self.do_average_in_cpu = do_average_in_cpu


class GradientClippingThreshold(OptimizerType):
    def extra_settings(self):
        default_gradient_clipping_threshold(self.threshold)

    def __init__(self, threshold):
        self.threshold = threshold

    def to_setting_kwargs(self):
        return dict()
