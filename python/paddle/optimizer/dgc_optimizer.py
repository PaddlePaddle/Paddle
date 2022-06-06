import paddle
from functools import reduce
from paddle import framework, _C_ops
from paddle.optimizer import Optimizer
from paddle.fluid import core
class DGCMomentumOptimizer(Optimizer):
    r"""
	:api_attr: Dygraph Graph

    DGC (Deep Gradient Compression) Momentum Optimizer. Original paper is https://arxiv.org/abs/1712.01887

    DGC reduces the communication bandwidth by sending only the important gradients (sparse update):\
        only gradients larger than a threshold are transmitted.

    To avoid losing information, DGC accumulates the rest of the gradients locally.

    Eventually, these gradients become large enough to be transmitted.

    Thus, DGC sends the large gradients immediately but eventually sends all of the gradients over time.

    To ensure no loss of accuracy, DGC employs momentum correction and local gradient clipping on top of the gradient sparsification to maintain model performance.

    DGC also uses momentum factor masking and warmup training to overcome the staleness problem caused by reduced communication.

    This optimizer will do two things:

        1. Compress the gradient by get TopK import value from tensor \
            and use it for allreduce to reduce network bandwidth.

        2. Call momentum to optimize the cost.

    Args:
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
            It can be a float value or any subclass of ``LRScheduler`` .
        momentum (float): Momentum factor. The default value is 0.9.
        rampup_begin_step (int): The beginning step from which gradient compression is implemented.
        rampup_step (int): Time steps used in sparsity warm-up periods. Default is 1.
            For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. \
                And when reach sparsity array ends, it will use 0.999 then and after.
        sparsity (list[float]): Get top important element from gradient tensor, the ratio is (1 - current sparsity). \
            Default is [0.999]. For example, if the sparsity is [0.99, 0.999], \
                the top [1%, 0.1%] important element will be transmitted.
        parameters (list|tuple, optional): List/Tuple of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. And you can specify different options for \
            different parameter groups such as the learning rate, weight decay, etc, \
            then the parameters are list of dict. Note that the learning_rate in paramter groups \
            represents the scale of base learning_rate. \
            The default value is None in static mode, at this time all parameters will be updated.
        use_nesterov (bool): Enables Nesterov momentum. True means use Nesterov. Default is False.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
            It canbe a float value as coeff of L2 regularization or \
            :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
            If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
            the regularization setting here in optimizer will be ignored for this parameter. \
            Otherwise, the regularization setting here in optimizer will take effect. \
            Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.

    Examples:
        .. code-block:: python

            import paddle
            optimizer = paddle.optimizer.DGCMomentumOptimizer(
                        learning_rate=0.0001,
                        momentum=0.9,
                        rampup_step=1000,
                        rampup_begin_step=1252,
                        sparsity=[0.999, 0.999])

    """
    _u_velocity_acc_str = "_dgc_u_"
    _v_velocity_acc_str = "_dgc_v_"

    def __init__(self,
                 learning_rate,
                 momentum,
                 rampup_begin_step,
                 rampup_step=1,
                 sparsity=[0.999],
                 parameters=None,
                 use_nesterov=False,
                 num_trainers=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        if not framework._non_static_mode():
            raise Exception("In static, please use 'paddle.fluid.optimizer.DGCMomentumOptimizer' for DGC.")

        assert core.is_compiled_with_cuda(), \
            "Paddle is not compiled with CUDA. DGC is only support GPU for now."

        assert learning_rate is not None
        assert momentum is not None
        super(DGCMomentumOptimizer, self).__init__(
            learning_rate=learning_rate,
            parameter_list=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.type = "dgc_momentum"
        self._momentum = momentum
        self._use_nesterov = bool(use_nesterov)

        assert rampup_begin_step >= 0, "rampup_begin_step must >= 0"
        self._rampup_begin_step = rampup_begin_step
        self._rampup_step = rampup_step
        self._sparsity = sparsity

        #self._rampup_begin_step_var = None
        #self._global_step_var = None

        self._dgc_clip_norm = None
        if grad_clip is not None:
            if not isinstance(grad_clip, paddle.nn.ClipGradByNorm):
                raise TypeError(
                    "The type of grad_clip should be 'ClipGradByNorm', because DGCMomentumOptimizer only support ClipGradByNorm"
                )
            assert isinstance(
                num_trainers, int
            ), "The type of num_trainers should be 'int', but received %s" % type(
                num_trainers)
            assert num_trainers > 0, "The value of num_trainers should be greater than 0!"

            self._num_trainers = num_trainers
            self._dgc_clip_norm = grad_clip.clip_norm * (num_trainers ** -0.5)

        self.regular_type, self.regular_coeff = self._get_regularization_param(
            weight_decay)
    
    def _get_regularization_param(self, regularization):
        regular_type = 0
        regular_coeff = 0.0

        if isinstance(regularization, float):
            regular_type = 2
            regular_coeff = regularization
            return regular_type, regular_coeff
        
        if regularization is not None:
            regular_coeff = regularization._regularization_coeff
            from paddle.regularizer import L1Decay, L2Decay
            if isinstance(regularization, L1Decay):
                regular_type = 1
            elif isinstance(regularization, L2Decay):
                regular_type = 2
            else:
                assert False, 'regularization must be None|L1Decay|L2Deacy'
        return regular_type, 
    
    def _is_use_dgc(self, param_var, grad_var):
        var_numel = abs(reduce(lambda x, y: x * y, param_var.shape))
        if var_numel < 16384 or \
           param_var.type == core.VarDesc.VarType.SELECTED_ROWS  or \
           grad_var.type == core.VarDesc.VarType.SELECTED_ROWS  or  \
               param_var.dtype != core.VarDesc.VarType.FP32 :
            return False
        return True
    
    def _apply_optimize(self, loss, startup_program, params_grads):
        """
        Second part of `minimize`, appending optimization operators for
        given `params_grads` pairs.
        Args:
            loss (Tensor): loss tensor to run optimizations.
            startup_program (Program): startup_program for initializing parameters
                in `parameters`.
            params_grads (list): list of (param, grad) pair to do optimization.
        Returns:
            list: A list of operators appended to the current program.
        """

        with program_guard(framework.default_main_program(),
                            framework.default_startup_program()):
            if isinstance(params_grads, list):
                if self._grad_clip is not None:
                    params_grads = self._grad_clip(params_grads)
                params_grads = self.append_regularization_ops(
                    params_grads, self.regularization)
            else:
                grad_clip = params_grads['grad_clip']
                if grad_clip is not None:
                    params_grads['params'] = grad_clip(params_grads[
                        'params'])

                params_grads['params'] = self.append_regularization_ops(
                    params_grads['params'], self.regularization)
            optimize_ops = self._create_optimization_pass(params_grads)


    

    

    
    
