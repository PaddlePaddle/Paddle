import py_paddle.swig_paddle as swig_api

import paddle.trainer_config_helpers.config_parser_utils as config_parser_utils
import paddle.trainer_config_helpers.optimizers as v1_optimizers
"""
Optimizers(update equation) for SGD method.

TODO(yuyang18): Complete comments.
"""

__all__ = [
    'Momentum', 'Adam', 'Adamax', 'AdaGrad', 'DecayedAdaGrad', 'AdaDelta',
    'RMSProp', 'ModelAverage', 'L2Regularization'
]


class Optimizer(object):
    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            del kwargs['batch_size']  # not important for python library.

        def __impl__():
            v1_optimizers.settings(batch_size=1, **kwargs)

        self.__opt_conf_proto__ = config_parser_utils.parse_optimizer_config(
            __impl__)
        self.__opt_conf__ = swig_api.OptimizationConfig.createFromProto(
            self.__opt_conf_proto__)

    def enable_types(self):
        """
        get enable_types for each optimizer.
        enable_types = [value, gradient, momentum, etc]
        For each optimizer(SGD, Adam), GradientMachine should enable different
        buffers.
        """
        tmp = swig_api.ParameterOptimizer.create(self.__opt_conf__)
        assert isinstance(tmp, swig_api.ParameterOptimizer)
        return tmp.getParameterTypes()

    def create_local_updater(self):
        return swig_api.ParameterUpdater.createLocalUpdater(self.__opt_conf__)

    def create_remote_updater(self, pass_num):
        return swig_api.ParameterUpdater.createRemoteUpdater(self.__opt_conf__,
                                                             pass_num)


class Momentum(Optimizer):
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

    def __init__(self, momentum=None, sparse=False, **kwargs):
        learning_method = v1_optimizers.MomentumOptimizer(
            momentum=momentum, sparse=sparse)
        super(Momentum, self).__init__(
            learning_method=learning_method, **kwargs)


class Adam(Optimizer):
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

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        learning_method = v1_optimizers.AdamOptimizer(
            beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(Adam, self).__init__(learning_method=learning_method, **kwargs)


class Adamax(Optimizer):
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

    def __init__(self, beta1=0.9, beta2=0.999, **kwargs):
        learning_method = v1_optimizers.AdamaxOptimizer(
            beta1=beta1, beta2=beta2)
        super(Adamax, self).__init__(learning_method=learning_method, **kwargs)


class AdaGrad(Optimizer):
    """
    Adagrad(for ADAptive GRAdient algorithm) optimizer.

    For details please refer this `Adaptive Subgradient Methods for
    Online Learning and Stochastic Optimization
    <http://www.magicbroom.info/Papers/DuchiHaSi10.pdf>`_.

    ..  math::

        G &= \\sum_{\\tau=1}^{t} g_{\\tau} g_{\\tau}^T \\\\
        w & = w - \\eta diag(G)^{-\\frac{1}{2}} \\circ g
    """

    def __init__(self, **kwargs):
        learning_method = v1_optimizers.AdaGradOptimizer()
        super(AdaGrad, self).__init__(learning_method=learning_method, **kwargs)


class DecayedAdaGrad(Optimizer):
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

    def __init__(self, rho=0.95, epsilon=1e-06, **kwargs):
        learning_method = v1_optimizers.DecayedAdaGradOptimizer(
            rho=rho, epsilon=epsilon)
        super(DecayedAdaGrad, self).__init__(
            learning_method=learning_method, **kwargs)


class AdaDelta(Optimizer):
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

    def __init__(self, rho=0.95, epsilon=1e-06, **kwargs):
        learning_method = v1_optimizers.AdaDeltaOptimizer(
            rho=rho, epsilon=epsilon)
        super(AdaDelta, self).__init__(
            learning_method=learning_method, **kwargs)


class RMSProp(Optimizer):
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

    def __init__(self, rho=0.95, epsilon=1e-6, **kwargs):
        learning_method = v1_optimizers.RMSPropOptimizer(
            rho=rho, epsilon=epsilon)
        super(RMSProp, self).__init__(learning_method=learning_method, **kwargs)


ModelAverage = v1_optimizers.ModelAverage
L2Regularization = v1_optimizers.L2Regularization

if __name__ == '__main__':
    swig_api.initPaddle('--use_gpu=false')
    for opt in [
            Momentum(), Adam(), Adamax(), AdaGrad(), DecayedAdaGrad(),
            AdaDelta(), RMSProp(), Adam(
                model_average=ModelAverage(average_window=0.5),
                regularization=L2Regularization(rate=0.5),
                gradient_clipping_threshold=25)
    ]:
        print opt, opt.enable_types()
