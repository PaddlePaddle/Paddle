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

from paddle.trainer.config_parser import Settings, default_decay_rate, \
    default_gradient_clipping_threshold, default_momentum

from .default_decorators import wrap_param_default

__all__ = [
    'Optimizer', 'BaseSGDOptimizer', 'MomentumOptimizer', 'AdamaxOptimizer',
    'AdamOptimizer', 'AdaGradOptimizer', 'RMSPropOptimizer',
    'DecayedAdaGradOptimizer', 'AdaDeltaOptimizer', 'BaseRegularization',
    'L2Regularization', 'settings', 'ModelAverage'
]


class Optimizer(object):
    def to_setting_kwargs(self):
        raise NotImplementedError()

    def extra_settings(self):
        pass

    @property
    def is_support_sparse(self):
        return True


class BaseSGDOptimizer(Optimizer):
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
        w & = w - \\frac{\\eta m(w, t)}{\\sqrt{v(w,t) + \\epsilon}}

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


class BaseRegularization(Optimizer):
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


class ModelAverage(Optimizer):
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


class GradientClippingThreshold(Optimizer):
    def extra_settings(self):
        default_gradient_clipping_threshold(self.threshold)

    def __init__(self, threshold):
        self.threshold = threshold

    def to_setting_kwargs(self):
        return dict()


def __extends__(dict1, dict2):
    for key in dict2:
        assert key not in dict1
        dict1[key] = dict2[key]
    return dict1


@wrap_param_default(
    ['learning_method'], default_factory=lambda _: MomentumOptimizer())
@wrap_param_default(
    ['regularization'], default_factory=lambda _: BaseRegularization())
def settings(batch_size,
             learning_rate=1e-3,
             learning_rate_decay_a=0.,
             learning_rate_decay_b=0.,
             learning_rate_schedule='poly',
             learning_rate_args='',
             async_lagged_grad_discard_ratio=1.5,
             learning_method=None,
             regularization=None,
             is_async=False,
             model_average=None,
             gradient_clipping_threshold=None):
    """
    Set the optimization method, learning rate, batch size, and other training
    settings. The currently supported algorithms are SGD and Async-SGD.

    ..  warning::

        Note that the 'batch_size' in PaddlePaddle is not equal to global
        training batch size. It represents the single training process's batch
        size. If you use N processes to train one model, for example use three
        GPU machines, the global batch size is N*'batch_size'.

    :param batch_size: batch size for one training process.
    :type batch_size: int
    :param learning_rate: learning rate for SGD
    :type learning_rate: float
    :param learning_method: The extension optimization algorithms of gradient
                            descent, such as momentum, adagrad, rmsprop, etc.
                            Note that it should be instance with base type
                            BaseSGDOptimizer.
    :type learning_method: BaseSGDOptimizer
    :param regularization: The regularization method.
    :type regularization: BaseRegularization
    :param is_async: Is Async-SGD or not. Default value is False.
    :type is_async: bool
    :param model_average: Model Average Settings.
    :type model_average: ModelAverage
    :param gradient_clipping_threshold: gradient clipping threshold. If gradient
                                        value larger than some value, will be
                                        clipped.
    :type gradient_clipping_threshold: float
    :param async_lagged_grad_discard_ratio: async SGD gradient commit control,
          when async_lagged_grad_discard_ratio * num_gradient_servers commit passed, 
          the current async SGD gradient is discarded.
    :type async_lagged_grad_discard_ratio: float
    """
    if isinstance(regularization, BaseRegularization):
        regularization = [regularization]

    assert isinstance(learning_method, Optimizer)
    if isinstance(learning_method, BaseSGDOptimizer):
        algorithm = 'async_sgd' if is_async else 'sgd'
    else:
        algorithm = 'owlqn'

    args = [
        'batch_size', 'learning_rate', 'learning_rate_decay_a',
        'learning_rate_decay_b', 'learning_rate_schedule', 'learning_rate_args',
        'gradient_clipping_threshold', 'async_lagged_grad_discard_ratio'
    ]
    kwargs = dict()
    kwargs['algorithm'] = algorithm
    for arg in args:
        kwargs[arg] = locals()[arg]

    kwargs = __extends__(kwargs, learning_method.to_setting_kwargs())
    learning_method.extra_settings()

    for regular in regularization:
        assert isinstance(regular, BaseRegularization)
        regular.algorithm = algorithm
        regular.learning_method = kwargs['learning_method']
        kwargs = __extends__(kwargs, regular.to_setting_kwargs())
        regular.extra_settings()

    if gradient_clipping_threshold is not None:
        gradient_clipping_threshold = GradientClippingThreshold(
            threshold=gradient_clipping_threshold)

    for each in [model_average, gradient_clipping_threshold]:
        if each is not None:
            assert isinstance(each, Optimizer)
            each.algorithm = algorithm
            each.learning_method = kwargs['learning_method']
            kwargs = __extends__(kwargs, each.to_setting_kwargs())
            each.extra_settings()

    # Do Check?
    Settings(**kwargs)
