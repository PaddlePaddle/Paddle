# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import copy
import json
import logging
import collections
import numpy as np

from .... import core
from ....framework import IrGraph
from . import utils
from .quantization_pass import QuantizationTransformPass
from .quantization_pass import QuantizationFreezePass
from .quantization_pass import ConvertToInt8Pass
from .quantization_pass import AddQuantDequantPass
from .quantization_pass import OutScaleForTrainingPass
from .quantization_pass import OutScaleForInferencePass
from .quantization_pass import QuantizationTransformPassV2
from .quantization_pass import QuantWeightPass
from .quantization_pass import AddQuantDequantPassV2
from .quantization_pass import AddQuantDequantForInferencePass
from .....static import global_scope
from .....static import CompiledProgram
from .....fluid.log_helper import get_logger

_logger = get_logger(__name__, level=logging.INFO)

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max',
    'channel_wise_abs_max',
    'range_abs_max',
    'moving_average_abs_max',
]
WEIGHT_QUANTIZATION_TYPES_TENSORRT = ['channel_wise_abs_max']

ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max',
    'range_abs_max',
    'moving_average_abs_max',
]

ACTIVATION_QUANTIZATION_TYPES_TENSORRT = [
    'range_abs_max',
    'moving_average_abs_max',
]

VALID_DTYPES = ['int8']

TRANSFORM_PASS_OP_TYPES = utils._weight_supported_quantizable_op_type
QUANT_DEQUANT_PASS_OP_TYPES = utils._act_supported_quantizable_op_type

TENSORRT_OP_TYPES = [
    'mul',
    'conv2d',
    'pool2d',
    'depthwise_conv2d',
    'elementwise_add',
    'leaky_relu',
]

VARS_MAPPING_TABLE = './mapping_table_for_saving_inference_model'

_quant_config_default = {
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # ops of name_scope in not_quant_pattern list, will not be quantized
    'not_quant_pattern': ['skip_quant'],
    # ops of type in quantize_op_types, will be quantized
    'quantize_op_types': ['conv2d', 'depthwise_conv2d', 'mul'],
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. defaulf is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # if True, 'quantize_op_types' will be TENSORRT_OP_TYPES
    'for_tensorrt': False,
    # if True, 'quantoze_op_types' will be TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    'is_full_quantize': False,
    # if True, use onnx format to quant.
    'onnx_format': False,
    # quant post to get initial scale for quant_aware
    'quant_post_first': False,
    # whether scale can be train
    'scale_trainable': True,
}


def load_dict():
    with open(VARS_MAPPING_TABLE, 'r') as file:
        data = file.read()
        data = json.loads(data)
        return data


def save_dict(table):
    with open(VARS_MAPPING_TABLE, 'w') as file:
        file.write(json.dumps(table))


def _parse_configs(user_config):
    """
    check if user's configs are valid.
    Args:
        user_config(dict): user's config.
    Return:
        configs(dict): final configs will be used.
    """

    configs = copy.deepcopy(_quant_config_default)
    configs.update(user_config)

    assert isinstance(configs['for_tensorrt'], bool) and isinstance(
        configs['is_full_quantize'], bool
    ), "'for_tensorrt' and 'is_full_quantize' must both be bool'"

    # check if configs is valid
    if configs['for_tensorrt']:
        weight_types = WEIGHT_QUANTIZATION_TYPES_TENSORRT
        activation_types = ACTIVATION_QUANTIZATION_TYPES_TENSORRT
        platform = 'TensorRT'
    else:
        weight_types = WEIGHT_QUANTIZATION_TYPES
        activation_types = WEIGHT_QUANTIZATION_TYPES
        platform = 'PaddleLite'
    assert (
        configs['weight_quantize_type'] in weight_types
    ), "Unknown weight_quantize_type: {}. {} only supports {} ".format(
        configs['weight_quantize_type'], platform, weight_types
    )

    assert (
        configs['activation_quantize_type'] in activation_types
    ), "Unknown activation_quantize_type: {}. {} only supports {}".format(
        configs['activation_quantize_type'], platform, activation_types
    )

    assert isinstance(
        configs['weight_bits'], int
    ), "weight_bits must be int value."

    assert (
        configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16
    ), "weight_bits should be between 1 and 16."

    assert isinstance(
        configs['activation_bits'], int
    ), "activation_bits must be int value."

    assert (
        configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16
    ), "activation_bits should be between 1 and 16."

    assert isinstance(
        configs['not_quant_pattern'], (list, str)
    ), "not_quant_pattern must be list or str"

    assert isinstance(
        configs['quantize_op_types'], list
    ), "quantize_op_types must be a list"

    if configs['for_tensorrt']:
        configs['quantize_op_types'] = TENSORRT_OP_TYPES
    elif configs['is_full_quantize']:
        configs['quantize_op_types'] = (
            TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
        )
    else:
        for op_type in configs['quantize_op_types']:
            assert (op_type in QUANT_DEQUANT_PASS_OP_TYPES) or (
                op_type in TRANSFORM_PASS_OP_TYPES
            ), "{} is not support, \
                        now support op types are {}".format(
                op_type, TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
            )

    assert isinstance(configs['dtype'], str), "dtype must be a str."

    assert configs['dtype'] in VALID_DTYPES, "dtype can only be " + " ".join(
        VALID_DTYPES
    )

    assert isinstance(
        configs['window_size'], int
    ), "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."

    assert isinstance(
        configs['moving_rate'], float
    ), "moving_rate must be float value, The decay coefficient of moving average, default is 0.9."

    return configs


def quant_aware(
    program,
    place,
    config=None,
    scope=None,
    for_test=False,
    weight_quantize_func=None,
    act_quantize_func=None,
    weight_preprocess_func=None,
    act_preprocess_func=None,
    optimizer_func=None,
    executor=None,
    return_program=False,
    calib_config={},
    draw_graph=False,
    return_scale_dict=False,
    scale_dict=None,
    model_type=None,
    pattern_ops=None,
):
    """Add quantization  and dequantization operators to "program"
    for quantization training or testing.
    Args:
        program(paddle.static.Program): training or testing ``program``.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
            the executor run on which device.
        config(dict, optional): configs for quantization. if None, will use default config.
            Default: None.
        scope(paddle.static.Scope): Scope records the mapping between variable names and variables,
            similar to brackets in programming languages. Usually users can use
            `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.
            When ``None`` will use `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_ .
            Default: ``None``.
        for_test(bool): If the 'program' parameter is a test program, this parameter should be set to ``True``.
            Otherwise, set to ``False``.Default: False
        weight_quantize_func(function): Function that defines how to quantize weight. Using this
                can quickly test if user's quantization method works or not. In this function, user should
                both define quantization function and dequantization function, that is, the function's input
                is non-quantized weight and function returns dequantized weight. If None, will use
                quantization op defined by 'weight_quantize_type'.
                Default is None.
        act_quantize_func(function): Function that defines how to quantize activation. Using this
                can quickly test if user's quantization method works or not. In this function, user should
                both define quantization and dequantization process, that is, the function's input
                is non-quantized activation and function returns dequantized activation. If None, will use
                quantization op defined by 'activation_quantize_type'.
                Default is None.
        weight_preprocess_func(function): Function that defines how to preprocess weight before quantization. Using this
                can quickly test if user's preprocess method works or not. The function's input
                is non-quantized weight and function returns processed weight to be quantized. If None, the weight will
                be quantized directly.
                Default is None.
        act_preprocess_func(function): Function that defines how to preprocess activation before quantization. Using this
                can quickly test if user's preprocess method works or not. The function's input
                is non-quantized activation and function returns processed activation to be quantized. If None, the activation will
                be quantized directly.
                Default is None.
        optimizer_func(function): Fuction return a optimizer. When 'is_test' is False and user want to use self-defined
            quantization function and preprocess function, this function must be set. Default is None.
        exe(paddle.static.Executor): If user want to use self-defined quantization function and preprocess function, exe must be set for
                initialization. Default is None.
        return_program(bool): If user want return value is a Program rather than Compiled Program, This argument should be set True.
                Default is False.
        draw_graph(bool): whether to draw graph when quantization is initialized. In order to prevent cycle,
                the ERNIE model needs to be set to True. Default is False.
        return_scale_dict(bool): If user want to return scale dict, model_type and pattern_ops, this argument should be set True.
                Default is False.
        scale_dict(dict): Use scale dict to initialize scales in program. Default is None.
        model_type(str): Model type can be 'transformer' or 'non-transformer'. If model type is transformer, patterns will be analyzed.
                Default is None.
        pattern_ops(dict): Pattern_ops contain pattern name and corresponding ops. Default is None.
    Returns:
        paddle.static.CompiledProgram | paddle.static.Program: Program with quantization and dequantization ``operators``
    """

    scope = global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("quant_aware config {}".format(config))

    skip_tensor_list = []
    same_scale_tensor_list = []
    if model_type == 'transformer':
        _logger.error(
            'paddle.fluid.contrib.slim.quantization.quant_aware does not support transformer model.'
        )

    is_test = True if for_test else not config['scale_trainable']
    if config['quant_post_first'] and for_test:
        if 'quantizable_op_type' not in calib_config:
            calib_config['quantizable_op_type'] = config['quantize_op_types']
        exe = paddle.static.Executor() if executor is None else executor
        post_training_quantization = PostTrainingQuantizationProgram(
            exe,
            program,
            freeze_model=False,
            skip_tensor_list=skip_tensor_list,
            same_scale_tensor_list=same_scale_tensor_list,
            batch_nums=10,
            scale_dict=scale_dict,
            return_graph=True,
            **calib_config
        )
        main_graph = post_training_quantization.quantize()
        scale_dict = post_training_quantization._scale_dict
    else:
        main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)
        transform_pass_ops = []
        quant_dequant_ops = []
        for op_type in config['quantize_op_types']:
            if op_type in TRANSFORM_PASS_OP_TYPES:
                transform_pass_ops.append(op_type)
            elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
                quant_dequant_ops.append(op_type)
        if len(transform_pass_ops) > 0:
            trannsform_func = (
                'QuantizationTransformPassV2'
                if config['onnx_format']
                else 'QuantizationTransformPass'
            )
            transform_pass = eval(trannsform_func)(
                scope=scope,
                place=place,
                weight_bits=config['weight_bits'],
                activation_bits=config['activation_bits'],
                activation_quantize_type=config['activation_quantize_type'],
                weight_quantize_type=config['weight_quantize_type'],
                window_size=config['window_size'],
                moving_rate=config['moving_rate'],
                quantizable_op_type=transform_pass_ops,
                skip_pattern=config['not_quant_pattern'],
                weight_quantize_func=weight_quantize_func,
                act_quantize_func=act_quantize_func,
                weight_preprocess_func=weight_preprocess_func,
                act_preprocess_func=act_preprocess_func,
                optimizer_func=optimizer_func,
                executor=executor,
                is_test=is_test,
            )

            transform_pass.apply(main_graph)

        if len(quant_dequant_ops) > 0:
            qdq_func = (
                'AddQuantDequantPassV2'
                if config['onnx_format']
                else 'AddQuantDequantPass'
            )
            quant_dequant_pass = eval(qdq_func)(
                scope=scope,
                place=place,
                moving_rate=config['moving_rate'],
                quant_bits=config['activation_bits'],
                skip_pattern=config['not_quant_pattern'],
                quantizable_op_type=quant_dequant_ops,
                is_test=is_test,
                scale_dict=scale_dict,
            )

            quant_dequant_pass.apply(main_graph)

    out_scale_training_pass = OutScaleForTrainingPass(
        scope=scope,
        place=place,
        moving_rate=config['moving_rate'],
        is_test=is_test,
        scale_dict=scale_dict,
    )

    out_scale_training_pass.apply(main_graph)

    if (
        weight_preprocess_func is not None or act_preprocess_func is not None
    ) and not for_test:
        _logger.info(
            "When a preprocess_func is used in quant_aware, Need to save a mapping table to match variable names in the convert phase."
        )
        _logger.info(
            "The mapping table is saved as '{}'.".format(VARS_MAPPING_TABLE)
        )
        save_dict(main_graph.out_node_mapping_table)

    # TDOD: remove it.
    if draw_graph:
        main_graph.draw('./', 'graph.pdf')

    if for_test or return_program:
        quant_program = main_graph.to_program()
    else:
        quant_program = CompiledProgram(main_graph.graph)

    if return_scale_dict:
        return quant_program, scale_dict, model_type, pattern_ops
    else:
        return quant_program


def convert(
    program,
    place,
    config=None,
    scope=None,
    save_int8=False,
    save_clip_ranges_path='./',
):
    """
    convert quantized and well-trained ``program`` to final  quantized
    ``program``that can be used to  save ``inference model``.

    Args:
        program(paddle.static.Program): quantized and well-trained ``test program``.
        place(paddle.CPUPlace or paddle.CUDAPlace): This parameter represents
                the executor run on which device.
        config(dict, optional): configs for convert. if set None, will use
                default config. It must be same with config that used in
                'quant_aware'. Default is None.
        scope(paddle.static.Scope, optional):  Scope records the mapping between
                variable names and variables, similar to brackets in
                programming languages. Usually users can use
                `paddle.static.global_scope <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_.
                When ``None`` will use
                `paddle.static.global_scope() <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/executor_cn/global_scope_cn.html>`_
                . Default: ``None``.
        save_int8: Whether to return ``program`` which model parameters'
                dtype is ``int8``. This parameter can only be used to
                get model size. Default: ``False``.
        save_clip_ranges_path: If config.onnx_format=True, quantization clip ranges will be saved locally.
    Returns:
        Tuple : freezed program which can be used for inference.
                when ``save_int8`` is False, return ``freezed_program(paddle.static.Program)``.
                when ``save_int8`` is True, return ``freezed_program(paddle.static.Program)``
                and ``freezed_program_int8(paddle.static.Program)``
    """
    scope = global_scope() if not scope else scope

    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("convert config {}".format(config))
    test_graph = IrGraph(core.Graph(program.desc), for_test=True)

    if config['onnx_format']:
        quant_weight_pass = QuantWeightPass(scope, place)
        quant_weight_pass.apply(test_graph)
        try:
            out_scale_infer_pass = AddQuantDequantForInferencePass(
                scope=scope, place=place, quant_bits=config['activation_bits']
            )
            out_scale_infer_pass.apply(test_graph)
        except:
            _logger.warning(
                "Unable to convert quant model with onnx_format=True, please update PaddlePaddle >= 2.4.0"
            )
    else:
        out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
        out_scale_infer_pass.apply(test_graph)
        # Freeze the graph after training by adjusting the quantize
        # operators' order for the inference.
        freeze_pass = QuantizationFreezePass(
            scope=scope,
            place=place,
            weight_bits=config['weight_bits'],
            activation_bits=config['activation_bits'],
            weight_quantize_type=config['weight_quantize_type'],
        )
        if os.path.exists(VARS_MAPPING_TABLE):
            test_graph.out_node_mapping_table = load_dict()
        freeze_pass.apply(test_graph)

    freezed_program = test_graph.to_program()

    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
        convert_int8_pass.apply(test_graph)
        freezed_program_int8 = test_graph.to_program()
        return freezed_program, freezed_program_int8
    else:
        return freezed_program
