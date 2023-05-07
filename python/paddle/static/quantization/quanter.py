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

import paddle
from paddle.framework import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import IrGraph
from paddle.static.quantization import WeightQuantization
from paddle.static.quantization import QuantizationTransformPass
from paddle.static.quantization import QuantizationFreezePass
from paddle.static.quantization import ConvertToInt8Pass
from paddle.static.quantization import PostTrainingQuantization
from paddle.static.quantization import AddQuantDequantPass
from paddle.static.quantization import OutScaleForTrainingPass
from paddle.static.quantization import OutScaleForInferencePass
from ..common import get_logger
from ..common.patterns import get_patterns
from ..common.patterns_common import has_trainable_var, get_weight
from ..core.graph_wrapper import GraphWrapper
_logger = get_logger(__name__, level=logging.INFO)

try:
    from paddle.static.quantization import QuantizationTransformPassV2
    from paddle.static.quantization import QuantWeightPass
    from paddle.static.quantization import AddQuantDequantPassV2
    from paddle.static.quantization import PostTrainingQuantizationProgram
    from paddle.static.quantization import AddQuantDequantForInferencePass
    from paddle.static.quantization import quant_config
except:
    _logger.warning(
        "Some functions failed to import, better to update PaddlePaddle to the latest develop version."
    )

WEIGHT_QUANTIZATION_TYPES = [
    'abs_max', 'channel_wise_abs_max', 'range_abs_max', 'moving_average_abs_max'
]
WEIGHT_QUANTIZATION_TYPES_TENSORRT = ['channel_wise_abs_max']

ACTIVATION_QUANTIZATION_TYPES = [
    'abs_max', 'range_abs_max', 'moving_average_abs_max'
]

ACTIVATION_QUANTIZATION_TYPES_TENSORRT = [
    'range_abs_max', 'moving_average_abs_max'
]

VALID_DTYPES = ['int8']
try:
    TRANSFORM_PASS_OP_TYPES = list(
        quant_config.SUPPORT_WEIGHT_QUANTIZATION_OP_DICT.keys())
    QUANT_DEQUANT_PASS_OP_TYPES = list(
        quant_config.SUPPORT_ACT_QUANTIZATION_OP_DICT.keys())
except:
    from paddle.static.quantization import utils
    TRANSFORM_PASS_OP_TYPES = utils._weight_supported_quantizable_op_type
    QUANT_DEQUANT_PASS_OP_TYPES = utils._act_supported_quantizable_op_type

TENSORRT_OP_TYPES = [
    'mul', 'conv2d', 'pool2d', 'depthwise_conv2d', 'elementwise_add',
    'leaky_relu'
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
    'onnx_format': True,
    # quant post to get initial scale for quant_aware
    'quant_post_first': False,
    # whether scale can be train
    'scale_trainable': True,
    # Deploy backend, it could be: None, TensorRT, MKLDNN, ARM
    'deploy_backend': None
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
        configs['is_full_quantize'],
        bool), "'for_tensorrt' and 'is_full_quantize' must both be bool'"

    # check if configs is valid
    if configs['for_tensorrt']:
        weight_types = WEIGHT_QUANTIZATION_TYPES_TENSORRT
        activation_types = ACTIVATION_QUANTIZATION_TYPES_TENSORRT
        platform = 'TensorRT'
    else:
        weight_types = WEIGHT_QUANTIZATION_TYPES
        activation_types = WEIGHT_QUANTIZATION_TYPES
        platform = 'PaddleLite'
    assert configs['weight_quantize_type'] in weight_types, \
        "Unknown weight_quantize_type: {}. {} only supports {} ".format(configs['weight_quantize_type'],
                platform, weight_types)

    assert configs['activation_quantize_type'] in activation_types, \
        "Unknown activation_quantize_type: {}. {} only supports {}".format(configs['activation_quantize_type'],
                platform, activation_types)

    assert isinstance(configs['weight_bits'], int), \
        "weight_bits must be int value."

    assert (configs['weight_bits'] >= 1 and configs['weight_bits'] <= 16), \
        "weight_bits should be between 1 and 16."

    assert isinstance(configs['activation_bits'], int), \
        "activation_bits must be int value."

    assert (configs['activation_bits'] >= 1 and configs['activation_bits'] <= 16), \
        "activation_bits should be between 1 and 16."

    assert isinstance(configs['not_quant_pattern'], (list, str)), \
        "not_quant_pattern must be list or str"

    assert isinstance(configs['quantize_op_types'], list), \
        "quantize_op_types must be a list"

    if configs['for_tensorrt']:
        configs['quantize_op_types'] = TENSORRT_OP_TYPES
    elif configs['is_full_quantize']:
        configs[
            'quantize_op_types'] = TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES
    else:
        for op_type in configs['quantize_op_types']:
            assert (op_type in QUANT_DEQUANT_PASS_OP_TYPES) or (
                op_type in TRANSFORM_PASS_OP_TYPES), "{} is not support, \
                        now support op types are {}".format(
                    op_type,
                    TRANSFORM_PASS_OP_TYPES + QUANT_DEQUANT_PASS_OP_TYPES)

    assert isinstance(configs['dtype'], str), \
        "dtype must be a str."

    assert (configs['dtype'] in VALID_DTYPES), \
        "dtype can only be " + " ".join(VALID_DTYPES)

    assert isinstance(configs['window_size'], int), \
        "window_size must be int value, window size for 'range_abs_max' quantization, default is 10000."

    assert isinstance(configs['moving_rate'], float), \
        "moving_rate must be float value, The decay coefficient of moving average, default is 0.9."

    deploy_backend = configs['deploy_backend']
    assert not deploy_backend or deploy_backend.lower() in [
        'tensorrt', 'mkldnn', 'arm'
    ], "Deploy Backend {} not support, please choose None, tensorrt or mkldnn.".format(
        deploy_backend)
    try:
        if not deploy_backend:
            configs['quant_config'] = quant_config.BaseQuantizer(
                quantizable_op_type=configs['quantize_op_types'],
                quant_bits=configs['weight_bits'], )
        elif deploy_backend.lower() == "tensorrt":
            configs['quant_config'] = quant_config.TensorRTQuantizer(
                quantizable_op_type=configs['quantize_op_types'],
                quant_bits=configs['weight_bits'], )
        elif deploy_backend.lower() == "mkldnn":
            configs['quant_config'] = quant_config.MKLDNNQuantizer(
                quantizable_op_type=configs['quantize_op_types'],
                quant_bits=configs['weight_bits'], )
        elif deploy_backend.lower() == "arm":
            configs['quant_config'] = quant_config.ARMCPUQuantizer(
                quantizable_op_type=configs['quantize_op_types'],
                quant_bits=configs['weight_bits'], )
    except:
        _logger.warning(
            "Set deploy_backend failed, Please update to PaddlePaddle Develop.")

    return configs


def quant_aware(program,
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
                pattern_ops=None):
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

    scope = paddle.static.global_scope() if not scope else scope
    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("quant_aware config {}".format(config))

    def find_next_ops(program, var_name):
        """
        Find all followed ops for the input variable.
        """
        block = program.global_block()
        res_ops = []
        for op in block.ops:
            if var_name in op.input_arg_names:
                res_ops.append(op)
        return res_ops

    def find_pre_ops(program, var_name):
        """
        Find all followed ops for the input variable.
        """
        block = program.global_block()
        res_ops = []
        for op in block.ops:
            if var_name in op.output_arg_names:
                res_ops.append(op)
        return res_ops

    def _is_skip_layernorm(program, op):
        if get_weight(op) is not None:
            return False

        output_names = op._op.output_arg_names
        for output_name in output_names:
            for next_op in find_next_ops(program, output_name):
                if next_op.type == 'layer_norm':
                    return True
        return False

    skip_tensor_list = []
    same_scale_tensor_list = []
    if model_type == 'transformer' and pattern_ops is None:
        pattern_ops, model_type = get_patterns(program)
        if model_type != 'transformer':
            _logger.info(
                'Warning! After analysis, the real model type is not transformer! If you encounter this situation, please raise an issue let us know in which case "get_patterns" determines model type is not transformer.'
            )
    if model_type == 'transformer':
        not_skip_quant_list = []
        for part_name, ops in pattern_ops.items():
            if 'MHA' in part_name:
                qkv_weight_tensor = []
                qkv_output_tensor = []
                ### get qkv
                output_names = ops[0]._op.output_arg_names
                for output_name in output_names:
                    for next_op in find_next_ops(program, output_name):
                        if next_op.type in ['mul', 'matmul_v2']:
                            qkv_weight_tensor.append(next_op.input('Y')[0])

                same_scale_tensor_list.append(qkv_weight_tensor)

                for op in ops:
                    if op._op.type in ['matmul', 'matmul_v2'] and (
                            not has_trainable_var(op)):
                        input_names = op._op.input_arg_names
                        for input_name in input_names:
                            pre_op = find_pre_ops(program, input_name)[0]
                            if pre_op.type == 'softmax' or pre_op.type == 'dropout':
                                continue
                            elif pre_op.type == 'scale':
                                qkv_output_tensor.append(
                                    input_name + '#/#{}'.format(
                                        pre_op.attr('scale')))
                            else:
                                qkv_output_tensor.append(input_name)
                    elif op._op.type == 'elementwise_add':
                        if _is_skip_layernorm(program, op):
                            not_skip_quant_list.append(op)
                same_scale_tensor_list.append(qkv_output_tensor)
            elif 'FFN' in part_name:
                for op in ops:
                    if op._op.type == 'elementwise_add':
                        if _is_skip_layernorm(program, op):
                            not_skip_quant_list.append(op)
        tmp_graph = GraphWrapper(program)
        for op in tmp_graph.ops():
            ### find elementwise_add in skip layernorm
            if op._op.type == 'elementwise_add' and op not in not_skip_quant_list:
                op._op._set_attr("op_namescope", "skip_quant")

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
            **calib_config)
        main_graph = post_training_quantization.quantize()
        scale_dict = post_training_quantization._scale_dict
        sub_graphs = [sub_graph for sub_graph in main_graph.all_sub_graphs()]
    else:
        main_graph = IrGraph(core.Graph(program.desc), for_test=for_test)
        sub_graphs = [sub_graph for sub_graph in main_graph.all_sub_graphs()]
        transform_pass_ops = []
        quant_dequant_ops = []
        if 'quant_config' in config and config['quant_config']:
            transform_pass_ops = config[
                'quant_config'].weight_quant_operation_types
            quant_dequant_ops = config[
                'quant_config'].activation_quant_operation_types
        else:
            for op_type in config['quantize_op_types']:
                if op_type in TRANSFORM_PASS_OP_TYPES:
                    transform_pass_ops.append(op_type)
                elif op_type in QUANT_DEQUANT_PASS_OP_TYPES:
                    quant_dequant_ops.append(op_type)
        if len(transform_pass_ops) > 0:
            trannsform_func = 'QuantizationTransformPassV2' if config[
                'onnx_format'] else 'QuantizationTransformPass'
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
                is_test=is_test)

            for sub_graph in sub_graphs:
                transform_pass.apply(sub_graph)

        if len(quant_dequant_ops) > 0:
            qdq_func = 'AddQuantDequantPassV2' if config[
                'onnx_format'] else 'AddQuantDequantPass'
            quant_dequant_pass = eval(qdq_func)(
                scope=scope,
                place=place,
                moving_rate=config['moving_rate'],
                quant_bits=config['activation_bits'],
                skip_pattern=config['not_quant_pattern'],
                quantizable_op_type=quant_dequant_ops,
                is_test=is_test)

            for sub_graph in sub_graphs:
                quant_dequant_pass.apply(sub_graph)

    out_scale_training_pass = OutScaleForTrainingPass(
        scope=scope,
        place=place,
        moving_rate=config['moving_rate'],
        is_test=is_test,
        scale_dict=scale_dict)

    for sub_graph in sub_graphs:
        out_scale_training_pass.apply(sub_graph)

    if (weight_preprocess_func is not None or act_preprocess_func is not None
        ) and not for_test and not config['onnx_format']:
        _logger.info(
            "When a preprocess_func is used in quant_aware, Need to save a mapping table to match variable names in the convert phase."
        )
        _logger.info("The mapping table is saved as '{}'.".format(
            VARS_MAPPING_TABLE))
        for sub_graph in sub_graphs:
            save_dict(sub_graph.out_node_mapping_table)

    # TDOD: remove it.
    if draw_graph:
        main_graph.draw('./', 'graph.pdf')

    if for_test or return_program:
        quant_program = main_graph.to_program()
    else:
        quant_program = paddle.static.CompiledProgram(main_graph.graph)

    if return_scale_dict:
        return quant_program, scale_dict, model_type, pattern_ops
    else:
        return quant_program


def quant_post_static(executor,
                      model_dir,
                      quantize_model_path,
                      batch_generator=None,
                      sample_generator=None,
                      data_loader=None,
                      model_filename=None,
                      params_filename=None,
                      save_model_filename='model.pdmodel',
                      save_params_filename='model.pdiparams',
                      batch_size=1,
                      batch_nums=None,
                      scope=None,
                      algo='hist',
                      round_type='round',
                      hist_percent=0.9999,
                      bias_correction=False,
                      quantizable_op_type=[
                          "conv2d", "depthwise_conv2d", "conv2d_transpose",
                          "mul", "matmul", "matmul_v2"
                      ],
                      is_full_quantize=False,
                      weight_bits=8,
                      activation_bits=8,
                      activation_quantize_type='range_abs_max',
                      weight_quantize_type='channel_wise_abs_max',
                      optimize_model=False,
                      onnx_format=True,
                      skip_tensor_list=None,
                      deploy_backend=None):
    """
    The function utilizes static post training quantization method to
    quantize the fp32 model. It uses calibrate data to calculate the
    scale factor of quantized variables, and inserts fake quantization
    and dequantization operators to obtain the quantized model.
    Args:
        executor(paddle.static.Executor): The executor to load, run and save the 
            quantized model.
        model_dir(str): The path of fp32 model that will be quantized, and 
            the model and params that saved by ``paddle.static.io.save_inference_model`` 
            are under the path.
        quantize_model_path(str): The path to save quantized model using api
            ``paddle.static.io.save_inference_model``.
        batch_generator(Python Generator): The batch generator provides 
                calibrate data for DataLoader, and it returns a batch every
                time. For sample_generator and batch_generator, only one
                can be set. Beisdes, batch_generator supports lod tensor.
        sample_generator(Python Generator): The sample generator provides 
            calibrate data for DataLoader, and it only returns a sample every time.
        data_loader(Python Generator, Paddle.io.DataLoader, optional): The
            Generator or Dataloader provides calibrate data, and it could
            return a batch every time.
        model_filename(str, optional): The name of model file. If parameters 
            are saved in separate files, set it as 'None'. Default: 'None'.
        params_filename(str, optional): The name of params file.
                When all parameters are saved in a single file, set it 
                as filename. If parameters are saved in separate files, 
                set it as 'None'. Default : 'None'.
        save_model_filename(str): The name of model file to save the quantized inference program.  Default: 'model.pdmodel'.
        save_params_filename(str): The name of file to save all related parameters. 
                If it is set None, parameters will be saved in separate files. Default: 'model.pdiparams'.
        batch_size(int, optional): The batch size of DataLoader, default is 1.
        batch_nums(int, optional): If batch_nums is not None, the number of calibrate 
                        data is 'batch_size*batch_nums'. If batch_nums is None, use all data
                        generated by sample_generator  as calibrate data.
        scope(paddle.static.Scope, optional): The scope to run program, use it to load 
                        and save variables. If scope is None, will use paddle.static.global_scope().
        algo(str, optional): If algo='KL', use KL-divergenc method to 
                        get the scale factor. If algo='hist', use the hist_percent of histogram 
                        to get the scale factor. If algo='mse', search for the best scale factor which
                        makes the mse loss minimal. Use one batch of data for mse is enough. If 
                        algo='avg', use the average of abs_max values  to get the scale factor. If 
                        algo='abs_max', use abs_max method to get the scale factor. Default: 'hist'.
        round_type(str, optional): The method of converting the quantized weights value
                        from float to int. Currently supports ['round', 'adaround'] methods.
                        Default is `round`, which is rounding nearest to the nearest whole number.
        hist_percent(float, optional): The percentile of histogram for algo hist.Default:0.9999.
        bias_correction(bool, optional): Bias correction method of https://arxiv.org/abs/1810.05723.
                        Default: False.
        quantizable_op_type(list[str], optional): The list of op types
                        that will be quantized. Default: ["conv2d", "depthwise_conv2d", 
                        "mul"].
        weight_bits(int, optional): quantization bit number for weights.
        activation_bits(int): quantization bit number for activation.
	activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This parameter only specifies the fake ops in quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. If it
                is 'abs_max', the scale will not be saved in fake ops.
        weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. Compared to 'abs_max',
                the model accuracy is usually higher when using 'channel_wise_abs_max'.
        is_full_quantize(bool): if True, apply quantization to all supported quantizable op type.
                        If False, only apply quantization to the input quantizable_op_type. Default is False.
        optimize_model(bool, optional): If set optimize_model as True, it applies some 
                passes to optimize the model before quantization. So far, the place of
                executor must be cpu it supports fusing batch_norm into convs.
        onnx_format(bool): Whether to export the quantized model with format of ONNX. Default is True.
        skip_tensor_list(list): List of skip quant tensor name.
        deploy_backend(str): Deploy backend, it could be None, TensorRT, MKLDNN, ARM.
                Other backends will continue to expand, the default is None, which means to
                use the default general quantization configuration.
    
    Returns:
        None
    """
    try:
        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            sample_generator=sample_generator,
            batch_generator=batch_generator,
            data_loader=data_loader,
            model_dir=model_dir.rstrip('/'),
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            scope=scope,
            algo=algo,
            round_type=round_type,
            hist_percent=hist_percent,
            bias_correction=bias_correction,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type=activation_quantize_type,
            weight_quantize_type=weight_quantize_type,
            onnx_format=onnx_format,
            skip_tensor_list=skip_tensor_list,
            optimize_model=optimize_model,
            deploy_backend=deploy_backend,  # support at Paddle develop
        )
    except:
        post_training_quantization = PostTrainingQuantization(
            executor=executor,
            sample_generator=sample_generator,
            batch_generator=batch_generator,
            data_loader=data_loader,
            model_dir=model_dir.rstrip('/'),
            model_filename=model_filename,
            params_filename=params_filename,
            batch_size=batch_size,
            batch_nums=batch_nums,
            scope=scope,
            algo=algo,
            round_type=round_type,
            hist_percent=hist_percent,
            bias_correction=bias_correction,
            quantizable_op_type=quantizable_op_type,
            is_full_quantize=is_full_quantize,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            activation_quantize_type=activation_quantize_type,
            weight_quantize_type=weight_quantize_type,
            onnx_format=onnx_format,
            skip_tensor_list=skip_tensor_list,
            optimize_model=optimize_model)

    post_training_quantization.quantize()
    post_training_quantization.save_quantized_model(
        quantize_model_path,
        model_filename=save_model_filename,
        params_filename=save_params_filename)


# We have changed the quant_post to quant_post_static.
# For compatibility, we keep quant_post api for now, and it will be
# deprecated in the future.
quant_post = quant_post_static


def convert(program, place, config=None, scope=None, save_int8=False):
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
    Returns:
        Tuple : freezed program which can be used for inference.
                when ``save_int8`` is False, return ``freezed_program(paddle.static.Program)``.
                when ``save_int8`` is True, return ``freezed_program(paddle.static.Program)``
                and ``freezed_program_int8(paddle.static.Program)``
    """
    scope = paddle.static.global_scope() if not scope else scope

    if config is None:
        config = _quant_config_default
    else:
        assert isinstance(config, dict), "config must be dict"
        config = _parse_configs(config)
    _logger.info("convert config {}".format(config))
    test_graph = IrGraph(core.Graph(program.desc), for_test=True)

    if config['onnx_format']:
        quant_weight_pass = QuantWeightPass(scope, place)
        for sub_graph in test_graph.all_sub_graphs():
            quant_weight_pass.apply(sub_graph)
        try:
            out_scale_infer_pass = AddQuantDequantForInferencePass(
                scope=scope, place=place, quant_bits=config['activation_bits'])
            for sub_graph in test_graph.all_sub_graphs():
                out_scale_infer_pass.apply(sub_graph)
        except:
            _logger.warning(
                "Unable to convert quant model with onnx_format=True, please update PaddlePaddle >= 2.4.0"
            )
    else:
        out_scale_infer_pass = OutScaleForInferencePass(scope=scope)
        for sub_graph in test_graph.all_sub_graphs():
            out_scale_infer_pass.apply(sub_graph)
        # Freeze the graph after training by adjusting the quantize
        # operators' order for the inference.
        freeze_pass = QuantizationFreezePass(
            scope=scope,
            place=place,
            weight_bits=config['weight_bits'],
            activation_bits=config['activation_bits'],
            weight_quantize_type=config['weight_quantize_type'])
        if os.path.exists(VARS_MAPPING_TABLE):
            test_graph.out_node_mapping_table = load_dict()
        for sub_graph in test_graph.all_sub_graphs():
            freeze_pass.apply(sub_graph)

    freezed_program = test_graph.to_program()

    # Move sub blocks persistable var to global block
    global_block = freezed_program.global_block()
    for _op in global_block.ops:
        if _op.type == "while":
            _block_id = _op.attr("sub_block").id
            _block = freezed_program.block(_block_id)
            persistables = []
            for _name, _var in _block.vars.items():
                if _var.persistable:
                    global_block._clone_variable(_var)
                    persistables.append(_name)
            for _name in persistables:
                _block._remove_var(_name)
            persistables.extend(_op.input('X'))
            _op.desc.set_input("X", persistables)

    assert not (
        save_int8 and config['onnx_format']
    ), "When onnx_format=True, already saved int8 weight,so you can't set save_int8=True."
    if save_int8:
        convert_int8_pass = ConvertToInt8Pass(scope=scope, place=place)
        for sub_graph in test_graph.all_sub_graphs():
            convert_int8_pass.apply(sub_graph)
        freezed_program_int8 = test_graph.to_program()
        return freezed_program, freezed_program_int8
    else:
        return freezed_program


def quant_post_dynamic(model_dir,
                       save_model_dir,
                       model_filename,
                       params_filename,
                       save_model_filename='model.pdmodel',
                       save_params_filename='model.pdiparams',
                       quantizable_op_type=["conv2d", "mul"],
                       weight_bits=8,
                       generate_test_model=False):
    '''
    The function utilizes static post training quantization method to
    quantize the fp32 model. In details, it quantizes the weight of some
    ops from float32 to int8/16. For the quantized model, there are two
    kinds of calculation method in the reference stage. Firstly, the
    quantized weight will be dequantized to float32, and then apply the
    float32 calculation. Secondly, collect the quantized scales of the
    inputs, and then apply the int8 calculation.
        
    Args:
        model_dir(str): The path of the fp32 model that will be quantized,
                and the model and params files are under the path.
        save_model_dir(str): The path to save the quantized model.
        model_filename(str): The name of file used to load the
                inference program. 
        params_filename(str): The name of file used to load all
                parameters.
        save_model_dir(str): The path used to save the quantized model.
        save_model_filename(str, optional): The name of file to 
                save the inference program. Default is 'model.pdmodel'.
        save_params_filename(str, optional): The name of file to 
                save all parameters. Default is 'model.pdiparams'.
        quantizable_op_type(list[str], optional): The list of ops 
                that will be quantized, and the quantized ops should be
                contained in ["conv2d", "depthwise_conv2d", "mul"]. 
                Default is ["conv2d", "depthwise_conv2d", "mul"].
        weight_bits(int, optional): The bits for the quantized weight, 
                and it should be 8 or 16. Default is 8.
        generate_test_model(bool, optional): If set generate_test_model 
                as True, it saves a fake quantized model, in which the weights 
                are quantized and dequantized. We can use PaddlePaddle to load 
                the fake quantized model and test the accuracy on GPU or CPU.
    '''

    weight_quant = WeightQuantization(
        model_dir=model_dir,
        model_filename=model_filename,
        params_filename=params_filename)

    weight_quant.quantize_weight_to_int(
        save_model_dir=save_model_dir,
        save_model_filename=save_model_filename,
        save_params_filename=save_params_filename,
        quantizable_op_type=quantizable_op_type,
        weight_bits=weight_bits,
        generate_test_model=generate_test_model)


# We have changed the quant_post_only_weight to quant_post_dynamic.
# For compatibility, we keep quant_post_only_weight api for now,
# and it will be deprecated in the future.
quant_post_only_weight = quant_post_dynamic


def pact(x, name=None):
    helper = LayerHelper("pact", **locals())
    dtype = 'float32'
    init_thres = 20
    u_param_attr = paddle.ParamAttr(
        name=x.name + '_pact',
        initializer=paddle.nn.initializer.Constant(value=init_thres),
        regularizer=paddle.regularizer.L2Decay(0.0001),
        learning_rate=1)
    u_param = helper.create_parameter(attr=u_param_attr, shape=[1], dtype=dtype)
    x = paddle.subtract(x,
                        paddle.nn.functional.relu(paddle.subtract(x, u_param)))
    x = paddle.add(x, paddle.nn.functional.relu(paddle.subtract(-u_param, x)))

    return x


def get_pact_optimizer():
    return paddle.optimizer.Momentum(0.0001, 0.9)
