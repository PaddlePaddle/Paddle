#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings

import paddle
from paddle.fluid import core
from paddle.fluid.framework import _dygraph_tracer, dygraph_only
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

from .amp_lists import black_list, white_list

AMP_RELATED_FLAGS = [
    'FLAGS_cudnn_exhaustive_search',
    'FLAGS_conv_workspace_size_limit',
    'FLAGS_cudnn_batchnorm_spatial_persistent',
]

AMP_RELATED_FLAGS_SETTING = {
    'FLAGS_cudnn_exhaustive_search': 1,
    'FLAGS_conv_workspace_size_limit': 1000,
    'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
}

AMP_LEVEL = core.AmpLevel
_g_amp_state_ = None


def amp_state():
    global _g_amp_state_
    return _g_amp_state_


class AMPGlobalState:
    def __init__(self):
        self.model_parameters = []
        self.use_master_grad = False
        self.already_register_final_backward_hook = False
        self.amp_dtype = 'float32'

    def __setattr__(self, name, val):
        self.__dict__[name] = val


_amp_global_state = AMPGlobalState()


def amp_global_state():
    return _amp_global_state


# NOTE(zhiqiu): similar as paddle.static.amp.fp16_lists.AutoMixedPrecisionLists._update_list
# The reason why not use AutoMixedPrecisionLists is that custom_black_varnames is not suitable for imperative mode.
def _update_list(
    custom_white_list, custom_black_list, level='O1', dtype='float16'
):
    """
    Update black and white list according to users' custom list.
    """
    if level == 'O0':
        _white_list = set()
        _black_list = set()
        return _white_list, _black_list
    _white_list = copy.copy(white_list()[dtype][level])
    _black_list = copy.copy(black_list()[dtype][level])
    if custom_white_list and custom_black_list:
        for op_name in custom_white_list:
            if op_name in custom_black_list:
                raise ValueError(
                    "Custom white list overlap " "custom black list"
                )
    if custom_white_list:
        for op_name in custom_white_list:
            if op_name in _black_list:
                _black_list.remove(op_name)
            _white_list.add(op_name)
    if custom_black_list:
        for op_name in custom_black_list:
            if op_name in _white_list:
                _white_list.remove(op_name)
            _black_list.add(op_name)
    return _white_list, _black_list


def _in_amp_guard():
    """
    Judge whether current code block is in `amp_guard` context.
    """
    tracer = _dygraph_tracer()
    if tracer:
        if tracer._amp_level == core.AmpLevel.O1:
            return True
        else:
            return False
    else:
        return False


def _in_pure_fp16_guard():
    tracer = _dygraph_tracer()
    return tracer and tracer._amp_level == core.AmpLevel.O2


def _is_gpu_float16_supported():
    """
    Judge whether current gpu support float16 amp.
    """
    prop = paddle.device.cuda.get_device_capability()
    return prop[0] >= 7


def _is_gpu_bfloat16_supported():
    """
    Judge whether current gpu support bfloat16 amp.
    """
    prop = paddle.device.cuda.get_device_capability()
    cuda_version = paddle.version.cuda()
    if cuda_version is not None and cuda_version != 'False':
        cuda_version_check = int(cuda_version.split('.')[0]) >= 11
    else:
        cuda_version_check = False
    return prop[0] >= 8 and cuda_version_check


def need_keep_fp32(layer, dtype):
    need_keep_fp32 = False
    # Highest prority. Because all the layers except BN will use bfloat16 params in bfoat16 training,
    # here we provide a option to keep fp32 param.
    if not layer._cast_to_low_precison:
        need_keep_fp32 = True
    # The BN layers will keep fp32
    elif isinstance(
        layer,
        (
            paddle.nn.BatchNorm,
            paddle.nn.BatchNorm1D,
            paddle.nn.BatchNorm2D,
            paddle.nn.BatchNorm3D,
            paddle.nn.SyncBatchNorm,
        ),
    ):
        need_keep_fp32 = True
    # layer._dtype is used to set params dtype. BF16 will use bf16 params.
    elif (layer._dtype == 'float16') or (
        (dtype == 'float16')
        and isinstance(
            layer,
            (
                paddle.nn.LayerNorm,
                paddle.nn.InstanceNorm1D,
                paddle.nn.InstanceNorm2D,
                paddle.nn.InstanceNorm3D,
            ),
        )
    ):
        need_keep_fp32 = True

    return need_keep_fp32


def set_excluded_layers(models, excluded_layers):
    excluded_layers_instances = []
    excluded_layers_types = []
    error_message = "excluded_layers must be either a nn.Layer instance/type or a list of nn.Layer instances/types."
    if excluded_layers is None:
        excluded_layers = []
    elif isinstance(excluded_layers, paddle.nn.Layer):
        excluded_layers_instances = [excluded_layers]
    elif isinstance(excluded_layers, type) and issubclass(
        excluded_layers, paddle.nn.Layer
    ):
        excluded_layers_types = [excluded_layers]
    elif isinstance(excluded_layers, list):
        for item in excluded_layers:
            if isinstance(item, paddle.nn.Layer):
                excluded_layers_instances.append(item)
            elif issubclass(item, paddle.nn.Layer):
                excluded_layers_types.append(item)
            else:
                raise TypeError(error_message)
    else:
        raise TypeError(error_message)

    for idx in range(len(excluded_layers_instances)):
        for layer in excluded_layers_instances[idx].sublayers(
            include_self=True
        ):
            layer._cast_to_low_precison = False
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if type(layer) in excluded_layers_types:
                layer._cast_to_low_precison = False


@dygraph_only
def amp_initialize(models, dtype, excluded_layers):
    set_excluded_layers(models, excluded_layers)
    for idx in range(len(models)):
        for layer in models[idx].sublayers(include_self=True):
            if need_keep_fp32(layer, dtype):
                continue
            if dtype == "float16" and isinstance(
                layer,
                (
                    paddle.incubate.nn.FusedFeedForward,
                    paddle.incubate.nn.FusedMultiHeadAttention,
                ),
            ):
                layer._amp_decorate(dtype=dtype)
                continue

            layer._to_impl(
                dtype=dtype, include_sublayers=False, floating_only=True
            )
    return models


def check_models(models):
    for model in models:
        if not isinstance(model, paddle.nn.Layer):
            raise RuntimeError(
                "Current train mode is pure fp16, models should be paddle.nn.Layer, but receive {}.".format(
                    type(model)
                )
            )
        if isinstance(model, paddle.DataParallel):
            raise RuntimeError(
                "For distributed AMP training, you should first use paddle.amp.decorate() to decotate origin model, and then call paddle.DataParallel get distributed model."
            )


def _is_valid_optimizer(optimizer):
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizer,
    )

    return isinstance(
        optimizer,
        (
            paddle.optimizer.Optimizer,
            paddle.fluid.optimizer.Optimizer,
            DygraphShardingOptimizer,
        ),
    )


def check_optimizers(optimizers):
    for optimizer in optimizers:
        if not _is_valid_optimizer(optimizer):
            raise RuntimeError(
                "Current train mode is pure fp16, optimizers should be paddle.optimizer.Optimizer or paddle.fluid.optimizer.Optimizer or DygraphShardingOptimizer, but receive {}.".format(
                    type(optimizer)
                )
            )


@signature_safe_contextmanager
@dygraph_only
def amp_guard(
    enable=True,
    custom_white_list=None,
    custom_black_list=None,
    level='O1',
    dtype='float16',
    use_promote=True,
):
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32 or float16) of each operator is decided
    by autocast algorithm for better performance.

    Commonly, it is used together with `GradScaler` to achieve Auto-Mixed-Precision in
    imperative mode. It is used together with `decorator` to achieve Pure fp16 in imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple, optional): The custom white_list. It's the set of ops that support
             fp16 calculation and are considered numerically-safe and performance-critical. These ops
             will be converted to fp16.
        custom_black_list(set|list|tuple, optional): The custom black_list. The set of ops that support fp16
             calculation and are considered numerically-dangerous and whose effects may also be
             observed in downstream ops. These ops will not be converted to fp16.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the input data type of each operator will be casted by white_list and black_list;
             O2 represent Pure fp16, all operators parameters and input data will be casted to fp16, except operators in black_list, don't support fp16 kernel and batchnorm. Default is O1(amp)
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.


    Examples:

     .. code-block:: python

        import numpy as np
        import paddle

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        conv2d = paddle.nn.Conv2D(3, 2, 3)
        data = paddle.to_tensor(data)
        with paddle.amp.amp_guard():
            conv = conv2d(data)
            print(conv.dtype) # FP16
        with paddle.amp.amp_guard(enable=False):
            conv = conv2d(data)
            print(conv.dtype) # FP32

    """
    amp_state = locals()
    global _g_amp_state_
    original_state = _g_amp_state_
    _g_amp_state_ = amp_state

    # check amp_level: O0-O2
    level = level.upper()
    if not (level in ['O0', 'OD', 'O1', 'O2']):
        raise ValueError("level should be O0, OD, O1 or O2.")

    # check amp_dtype: float16 or bfloat16
    dtype = dtype.lower()
    if enable:
        if not (dtype in ['float16', 'bfloat16']):
            raise ValueError(
                "If enable amp, dtype should be 'float16' or 'bfloat16'."
            )

    # check tracer
    tracer = _dygraph_tracer()
    if not tracer:
        raise ValueError(
            "current_tracer is None, maybe it is not in imperative mode."
        )

    # check device_type:
    # NOTE: Now, amp only support gpu for float16 and bfloat16, xpu for float16, npu for float16.
    # Maybe we will support cpu for bfloat16.
    if enable and not (
        tracer._expected_place.is_gpu_place()
        or tracer._expected_place.is_xpu_place()
        or tracer._expected_place.is_custom_place()
    ):
        warnings.warn(
            'amp_guard can only be enabled on CUDAPlace, XPUPlace, and CustomPlace, current place is %s, so it makes no effect.'
            % tracer._expected_place
        )
        enable = False
    # For xpu:
    if tracer._expected_place.is_xpu_place() and (dtype == 'bfloat16'):
        warnings.warn('XPUPlace only support float16 amp.')
        enable = False
    # For custom device:
    if tracer._expected_place.is_custom_place() and (dtype == 'bfloat16'):
        warnings.warn('CustomPlace only support float16 amp.')
        enable = False
    # For gpu float16: Compute Capability should >= 7.
    # For gpu bfloat16: Compute Capability should >= 8 & CUDA Version should >= 11.
    if tracer._expected_place.is_gpu_place():
        if (dtype == 'float16') and not _is_gpu_float16_supported():
            prop = paddle.device.cuda.get_device_capability()
            warnings.warn(
                "For float16, amp only support NVIDIA GPU with Compute Capability 7.0 or higher, current GPU is: %s, with Compute Capability: %d.%d."
                % (paddle.device.cuda.get_device_name(), prop[0], prop[1])
            )
        elif (dtype == 'bfloat16') and not _is_gpu_bfloat16_supported():
            prop = paddle.device.cuda.get_device_capability()
            cuda_version = paddle.version.cuda()
            warnings.warn(
                "For bfloat16, amp only support NVIDIA GPU with Compute Capability 8.0 or higher and CUDA Version 11.0 or higher, current GPU is: %s, with Compute Capability: %d.%d, current CUDA Version is: %s."
                % (
                    paddle.device.cuda.get_device_name(),
                    prop[0],
                    prop[1],
                    cuda_version,
                )
            )

    amp_dtype = dtype
    amp_global_state().amp_dtype = amp_dtype

    if level == 'OD':
        amp_level = AMP_LEVEL.OD
    elif level == 'O1':
        amp_level = AMP_LEVEL.O1
    elif level == 'O2':
        amp_level = AMP_LEVEL.O2
    elif level == 'O0':
        amp_level = AMP_LEVEL.O0

    _white_list, _black_list = _update_list(
        custom_white_list, custom_black_list, level, dtype
    )

    if not enable:
        amp_level = AMP_LEVEL.O0
        amp_dtype = "float32"

    # master_grad_hook will run at the end of backward.
    # Since backward_final_hook will be cleared once they have been
    # done, we should register the hook every step.
    if (
        amp_global_state().use_master_grad
        and not amp_global_state().already_register_final_backward_hook
    ):

        def master_grad_hook():
            core.eager.set_master_grads(amp_global_state().model_parameters)
            amp_global_state().already_register_final_backward_hook = False

        core.eager._add_backward_final_hook(master_grad_hook)
        amp_global_state().already_register_final_backward_hook = True

    if tracer:
        # enable auto_cast
        original_amp_level = tracer._amp_level
        tracer._amp_level = amp_level

        # set amp op list
        original_white_list, original_black_list = tracer._get_amp_op_list()
        tracer._set_amp_op_list(_white_list, _black_list)

        # TODO(zhiqiu) set amp related flags automatically in this guard
        # Currently, if FLAGS_cudnn_batchnorm_spatial_persistent is set True in amp_guard,
        # batch_norm can run in fast mode, but batch_norm_grad can not if backward if not executed insise amp_guard.
        # So, users need to set related flags manually.

        # original_flags = get_flags(AMP_RELATED_FLAGS)
        # set_flags(AMP_RELATED_FLAGS_SETTING)

        # set amp dtype
        original_amp_dtype = tracer._amp_dtype
        tracer._amp_dtype = amp_dtype

        # switch promote
        if amp_level == AMP_LEVEL.O2:
            original_use_promote = tracer._use_promote
            tracer._use_promote = use_promote

    # restore status
    try:
        yield
    finally:
        if tracer:
            _g_amp_state_ = original_state
            tracer._amp_level = original_amp_level
            tracer._set_amp_op_list(original_white_list, original_black_list)
            # set_flags(original_flags)
            tracer._amp_dtype = original_amp_dtype
            if amp_level == AMP_LEVEL.O2:
                tracer._use_promote = original_use_promote


class StateDictHook:
    def __init__(self, save_dtype):
        self._save_dtype = save_dtype

    def __call__(self, state_dict):
        for key in state_dict:
            param = state_dict[key]
            if paddle.is_floating_point(param):
                param_applied = paddle.cast(param, self._save_dtype)
                param_applied.name = param.name
                state_dict[key] = param_applied


def _set_multi_precision(optimizer, multi_precision):
    from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
        DygraphShardingOptimizer,
    )

    optimizer = (
        optimizer._inner_optimizer
        if isinstance(optimizer, DygraphShardingOptimizer)
        else optimizer
    )
    if hasattr(optimizer, "_multi_precision"):
        optimizer._multi_precision = multi_precision


@dygraph_only
def amp_decorate(
    models,
    optimizers=None,
    level='O1',
    dtype='float16',
    master_weight=None,
    save_dtype=None,
    master_grad=False,
    excluded_layers=None,
):
    """
    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.
    When level is O2(pure fp16), the decorate will cast all parameters of models to FP16, except BatchNorm, InstanceNorm and LayerNorm.

    Commonly, it is used together with `amp_guard` to achieve Pure fp16 in imperative mode.

    Args:
        models(Layer|list of Layer, optional): The defined models by user, models must be either a single model or a list of models. Default is None.
        optimizers(Optimizer|list of Optimizer, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are "O1" and "O2": O1 represent mixed precision, the decorator will do nothing;
             O2 represent Pure fp16/bf16, the decorator will cast all parameters of models to FP16/BF16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        master_weight(bool, optinal): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.
        save_dtype(float, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.
             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.

    Examples:

     .. code-block:: python

        # required: gpu
        # Demo1: single model and optimizer:
        import paddle

        model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer = paddle.optimizer.SGD(parameters=model.parameters())

        model, optimizer = paddle.amp.amp_decorate(models=model, optimizers=optimizer, level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = model(data)
            print(output.dtype) # FP16

        # required: gpu
        # Demo2: multi models and optimizers:
        model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

        models, optimizers = paddle.amp.amp_decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = models[0](data)
            output2 = models[1](data)
            print(output.dtype) # FP16
            print(output2.dtype) # FP16

        # required: gpu
        # Demo3: optimizers is None:
        model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer3 = paddle.optimizer.Adam(parameters=model2.parameters())

        model = paddle.amp.amp_decorate(models=model3, level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.amp_guard(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = model(data)
            print(output.dtype) # FP16
    """
    if not (level in ['O1', 'O2']):
        raise ValueError(
            "level should be O1 or O2, O1 represent AMP train mode, O2 represent Pure fp16 train mode."
        )
    if not (dtype in ['float16', 'bfloat16']):
        raise ValueError("dtype only support float16 or bfloat16.")

    if level == 'O1':
        if optimizers is None:
            return models
        else:
            return models, optimizers

    models_is_list = False
    if isinstance(models, paddle.nn.Layer):
        models_is_list = False
        models = [models]
        check_models(models)
    elif isinstance(models, list):
        check_models(models)
        models_is_list = True
    else:
        raise TypeError(
            "models must be either a single model or a list of models."
        )

    # initialize parameters of the model.
    amp_initialize(models=models, dtype=dtype, excluded_layers=excluded_layers)

    if optimizers is not None:
        # check optimizers
        optimizers_is_list = False
        if _is_valid_optimizer(optimizers):
            optimizers_is_list = False
            optimizers = [optimizers]
            check_optimizers(optimizers)
        elif isinstance(optimizers, list):
            check_optimizers(optimizers)
            optimizers_is_list = True
        else:
            raise TypeError(
                "optimizers must be either a single optimizer or a list of optimizers."
            )
        # support master_weight
        use_multi_precision = not (master_weight is False)
        for opt in optimizers:
            _set_multi_precision(opt, use_multi_precision)

        # support master_grad
        if master_grad:
            amp_global_state().use_master_grad = True
            for idx in range(len(models)):
                amp_global_state().model_parameters.extend(
                    models[idx].parameters()
                )

    if save_dtype is not None:
        if not (save_dtype in ['float16', 'bfloat16', 'float32', 'float64']):
            raise ValueError(
                "save_dtype can only be float16 float32 or float64, but your input save_dtype is %s."
                % save_dtype
            )
        for idx in range(len(models)):
            for layer in models[idx].sublayers(include_self=True):
                layer.register_state_dict_hook(StateDictHook(save_dtype))

    if models_is_list:
        if optimizers is not None:
            if optimizers_is_list:
                return models, optimizers
            else:
                return models, optimizers[0]
        else:
            return models
    else:
        if optimizers is not None:
            if optimizers_is_list:
                return models[0], optimizers
            else:
                return models[0], optimizers[0]
        else:
            return models[0]


def auto_cast(
    enable=True,
    custom_white_list=None,
    custom_black_list=None,
    level='O1',
    dtype='float16',
    use_promote=True,
):
    """
    Create a context which enables auto-mixed-precision(AMP) of operators executed in dynamic graph mode.
    If enabled, the input data type (float32, float16 or bfloat16) of each operator is decided
    by autocast algorithm for better performance.

    Commonly, it is used together with `GradScaler` and `decorator` to achieve Auto-Mixed-Precision in
    imperative mode.

    Args:
        enable(bool, optional): Enable auto-mixed-precision or not. Default is True.
        custom_white_list(set|list|tuple, optional): A default white list is already set. Usually there is no need to set custom white list.
             The set of ops should be considered numerically-safe and performance-critical. These ops will be converted to float16/bfloat16.
        custom_black_list(set|list|tuple, optional): A default black list is already set. You can set a custom black list according to the model.
             The set of ops are considered numerically-dangerous and whose effects may also be observed in downstream ops. These ops will not be
             converted to float16/bfloat16.
        level(str, optional): Auto mixed precision level. Accepted values are "O1", "O2" and "OD": At the O1 level, operators in the white list
             will use float16/bfloat16 inputs for calculations, and operators in the black list will use float32 inputs for calculations. At the O2
             level, model's parameters will be casted to float16/bfloat16 by using `decorator`, and operators that have all float16/bfloat16 inputs
             will be converted to float16/bfloat16, and that have any float32 input will be converted to float32. For the OD level, operators in
             default white list will compute in float16/bfloat16, and the others will compute in float32. Default is O1.
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        use_promote(bool, optional): Whether to promotes to fp32 when op has any float32 inputs. It is only supported when amp level is O2. Default is True.

    Examples:

     .. code-block:: python

        import paddle

        conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast():
            conv = conv2d(data)
            print(conv.dtype) # paddle.float16

        with paddle.amp.auto_cast(enable=False):
            conv = conv2d(data)
            print(conv.dtype) # paddle.float32

        with paddle.amp.auto_cast(custom_black_list={'conv2d'}):
            conv = conv2d(data)
            print(conv.dtype) # paddle.float32

        a = paddle.rand([2,3])
        b = paddle.rand([2,3])
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}):
            c = a + b
            print(c.dtype) # paddle.float16

        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O2'):
            d = a + b
            print(d.dtype) # paddle.float16

    """
    return amp_guard(
        enable, custom_white_list, custom_black_list, level, dtype, use_promote
    )


def decorate(
    models,
    optimizers=None,
    level='O1',
    dtype='float16',
    master_weight=None,
    save_dtype=None,
    master_grad=False,
    excluded_layers=None,
):
    """
    Decorate models and optimizers for auto-mixed-precision. When level is O1(amp), the decorate will do nothing.
    When level is O2(pure float16/bfloat16), the decorate will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm.

    Commonly, it is used together with `auto_cast` to achieve Pure float16/bfloat16 in imperative mode.

    Args:
        models(Layer|list of Layer): The defined models by user, models must be either a single model or a list of models. Default is None.
        optimizers(Optimizer|list of Optimizer, optional): The defined optimizers by user, optimizers must be either a single optimizer or a list of optimizers. Default is None.
        level(str, optional): Auto mixed precision level. Accepted values are 'O1' and 'O2': O1 represent mixed precision, the decorator will do nothing;
             O2 represent Pure float16/bfloat16, the decorator will cast all parameters of models to float16/bfloat16, except BatchNorm, InstanceNorm and LayerNorm. Default is O1(amp)
        dtype(str, optional): Whether to use 'float16' or 'bfloat16'. Default is 'float16'.
        master_weight(bool, optinal): For level='O2', whether to use multi-precision during weight updating. If master_weight is None, in O2 level optimizer will use multi-precision. Default is None.
        save_dtype(float, optional): The save model parameter dtype when use `paddle.save` or `paddle.jit.save`,it should be float16, bfloat16, float32, float64 or None.
             The save_dtype will not change model parameters dtype, it just change the state_dict dtype. When save_dtype is None, the save dtype is same as model dtype. Default is None.
        master_grad(bool, optional): For level='O2', whether to use float32 weight gradients for calculations such as gradient clipping, weight decay, and weight updates. If master_grad is enabled, the weight
             gradients will be float32 dtype after the backpropagation. Default is False, there is only float16 weight gradients.
        excluded_layers(Layer|list of Layer, optional): Specify the layers not to be decorated. The weights of these layers will always keep float32 when level is O2. `excluded_layers` can be specified as
             an Layer instance/type or a list of Layer instances/types. Default is None, the weights of the whole model will be casted to float16 or bfloat16.

    Examples:

     .. code-block:: python

        # required: gpu
        # Demo1: single model and optimizer:
        import paddle

        model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer = paddle.optimizer.SGD(parameters=model.parameters())

        model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = model(data)
            print(output.dtype) # FP16

        # required: gpu
        # Demo2: multi models and optimizers:
        model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer2 = paddle.optimizer.Adam(parameters=model2.parameters())

        models, optimizers = paddle.amp.decorate(models=[model, model2], optimizers=[optimizer, optimizer2], level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = models[0](data)
            output2 = models[1](data)
            print(output.dtype) # FP16
            print(output2.dtype) # FP16

        # required: gpu
        # Demo3: optimizers is None:
        model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        optimizer3 = paddle.optimizer.Adam(parameters=model3.parameters())

        model = paddle.amp.decorate(models=model3, level='O2')

        data = paddle.rand([10, 3, 32, 32])

        with paddle.amp.auto_cast(enable=True, custom_white_list=None, custom_black_list=None, level='O2'):
            output = model(data)
            print(output.dtype) # FP16
    """
    return amp_decorate(
        models,
        optimizers,
        level,
        dtype,
        master_weight,
        save_dtype,
        master_grad,
        excluded_layers,
    )
