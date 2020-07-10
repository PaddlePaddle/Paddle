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
import math
import os
import re
import logging
import numpy as np
from .... import io
from .... import core
from .... import framework
from ....executor import global_scope, Executor
from ....framework import IrGraph
from ....log_helper import get_logger
from .quantization_pass import QuantizationTransformPass
from .quantization_pass import QuantizationFreezePass
from .quantization_pass import AddQuantDequantPass
from .quantization_pass import _out_scale_op_list
from .quantization_pass import _get_op_input_var_names
from .quantization_pass import _get_op_output_var_names
from .quantization_pass import _get_output_name_index

__all__ = ['PostTrainingQuantization', 'WeightQuantization']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, \
        "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


def _set_variable_data(scope, place, var_name, np_value):
    '''
    Set the value of var node by name, if the node exits,
    '''
    assert isinstance(np_value, np.ndarray), \
        'The type of value should be numpy array.'
    var_node = scope.find_var(var_name)
    if var_node != None:
        tensor = var_node.get_tensor()
        tensor.set(np_value, place)


def _all_persistable_var_names(program):
    persistable_var_names = []
    for var in program.list_vars():
        if var.persistable:
            persistable_var_names.append(var.name)
    return persistable_var_names


def _remove_unused_var_nodes(graph):
    all_used_vars = set()
    ops = graph.all_op_nodes()
    for op_node in ops:
        for input_node in op_node.inputs:
            all_used_vars.add(input_node)
        for output_node in op_node.outputs:
            all_used_vars.add(output_node)

    all_used_vars = {n.node for n in all_used_vars}
    all_unused_vars = {
        n
        for n in filter(lambda node: node.node not in all_used_vars,
                        graph.all_var_nodes())
    }
    graph.safe_remove_nodes(all_unused_vars)
    return graph


def _remove_ctrl_vars(graph):
    remove_ctr_vars = set()
    for node in graph.all_var_nodes():
        if node.is_ctrl_var():
            remove_ctr_vars.add(node)
    graph.safe_remove_nodes(remove_ctr_vars)
    return graph


def _apply_pass(scope,
                graph,
                pass_name,
                attrs=None,
                attr_values=None,
                debug=False):
    ir_pass = core.get_pass(pass_name)
    cpp_graph = graph.graph
    if not cpp_graph.has('__param_scope__'):
        cpp_graph.set_not_owned('__param_scope__', scope)
    if attrs:
        assert attr_values and len(attrs) == len(
            attr_values), "Different number of pass attributes and their values."
        for attr, value in zip(attrs, attr_values):
            ir_pass.set(attr, value)
    ir_pass.apply(cpp_graph)
    if debug:
        graph.draw('.', 'qat_fp32_{}'.format(pass_name), graph.all_op_nodes())
    _remove_unused_var_nodes(graph)
    return graph


class PostTrainingQuantization(object):
    """
    Utilizing post training quantization methon to quantize the FP32 model,
    and it uses calibrate data to get the quantization information for all 
    quantized variables.
    """

    def __init__(self,
                 executor=None,
                 scope=None,
                 model_dir=None,
                 model_filename=None,
                 params_filename=None,
                 batch_generator=None,
                 sample_generator=None,
                 batch_size=10,
                 batch_nums=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 activation_bits=8,
                 weight_bits=8,
                 activation_quantize_type='range_abs_max',
                 weight_quantize_type='channel_wise_abs_max',
                 optimize_model=False,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        '''
        Constructor.

        Args:
            executor(fluid.Executor): The executor to load, run and save the
                quantized model.
            scope(fluid.Scope, optional): The scope of the program, use it to load 
                and save variables. If scope=None, get scope by global_scope(). 
            model_dir(str): The path of the fp32 model that will be quantized, 
                and the model and params files are under the path.
            model_filename(str, optional): The name of file to load the inference 
                program. If it is None, the default filename '__model__' will 
                be used. Default is 'None'.
            params_filename(str, optional): The name of file to load all parameters.
                When all parameters were saved in a single binary file, set it 
                as the real filename. If parameters were saved in separate files, 
                set it as 'None'. Default is 'None'.
            batch_generator(Python Generator): The batch generator provides 
                calibrate data for DataLoader, and it returns a batch every
                time. Note that, sample_generator and batch_generator, only one
                should be set. Beisdes, batch_generator supports lod tensor.
            sample_generator(Python Generator): The sample generator provides
                calibrate data for DataLoader, and it only returns a sample every
                time. Note that, sample_generator and batch_generator, only one
                should be set. Beisdes, sample_generator dose not support lod tensor.
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of 
                calibrate data is batch_size*batch_nums. If batch_nums is None, use 
                all data provided by sample_generator as calibrate data.
            algo(str, optional): If algo='KL', use KL-divergenc method to
                get the KL threshold for quantized activations and get the abs_max
                value for quantized weights. If algo='abs_max', get the abs max 
                value for activations and weights. If algo= 'min_max', get the min 
                and max value for quantized activations and weights. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops 
                that will be quantized. Default is ["conv2d", "depthwise_conv2d", 
                "mul"].
            is_full_quantized(bool, optional): If set is_full_quantized as True, 
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type 
                according to the input quantizable_op_type.
            activation_bits(int): quantization bit number for activation.
            weight_bits(int, optional): quantization bit number for weights.
            activation_quantize_type(str): quantization type for activation,
                now support 'range_abs_max', 'moving_average_abs_max' and 'abs_max'.
                This param only specifies the fake ops in saving quantized model.
                If it is 'range_abs_max' or 'moving_average_abs_max', we save the scale
                obtained by post training quantization in fake ops. Note that, if it
                is 'abs_max', the scale will not be saved in fake ops.
            weight_quantize_type(str): quantization type for weights,
                support 'abs_max' and 'channel_wise_abs_max'. This param only specifies
                the fake ops in saving quantized model, and we save the scale obtained
                by post training quantization in fake ops. Compared to 'abs_max',
                the model accuracy is usually higher when it is 'channel_wise_abs_max'.
            optimize_model(bool, optional): If set optimize_model as True, it applies
                some passes to the model before quantization, and it supports
                `conv2d/depthwise_conv2d + bn` pass so far. Some targets require the
                weights are quantized by tensor-wise method, which means the weights
                scale for all channel are the same. However, if fuse
                `conv2d/depthwise_conv2d + bn`, the weights scale for all channel will
                be different. In address this problem, fuse the pattern before
                quantization. Default False.
            is_use_cache_file(bool, optional): If set is_use_cache_file as False,
                all temp data will be saved in memory. If set is_use_cache_file as True,
                it will save temp data to disk. When the fp32 model is complex or
                the number of calibrate data is large, we should set is_use_cache_file
                as True. Defalut is False.
            cache_dir(str, optional): When is_use_cache_file is True, set cache_dir as
                the directory for saving temp data. Default is ./temp_post_training.
        Returns:
            None

        Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
            
            exe = fluid.Executor(fluid.CPUPlace())
            model_dir = path/to/fp32_model_params
            # set model_filename as None when the filename is __model__, 
            # otherwise set it as the real filename
            model_filename = None 
            # set params_filename as None when all parameters were saved in 
            # separate files, otherwise set it as the real filename
            params_filename = None
            save_model_path = path/to/save_model_path
            # prepare the sample generator according to the model, and the 
            # sample generator must return a sample every time. The reference
            # document: https://www.paddlepaddle.org.cn/documentation/docs/zh
            # /user_guides/howto/prepare_data/use_py_reader.html
            sample_generator = your_sample_generator
            batch_size = 10
            batch_nums = 10
            algo = "KL"
            quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]
            ptq = PostTrainingQuantization(
                        executor=exe,
                        sample_generator=sample_generator,
                        model_dir=model_dir,
                        model_filename=model_filename,
                        params_filename=params_filename,
                        batch_size=batch_size,
                        batch_nums=batch_nums,
                        algo=algo,
                        quantizable_op_type=quantizable_op_type)
            ptq.quantize()
            ptq.save_quantized_model(save_model_path)
        '''

        self._support_activation_quantize_type = [
            'range_abs_max', 'moving_average_abs_max', 'abs_max'
        ]
        self._support_weight_quantize_type = ['abs_max', 'channel_wise_abs_max']
        self._support_algo_type = ['KL', 'abs_max', 'min_max']
        self._support_quantize_op_type = \
            list(set(QuantizationTransformPass._supported_quantizable_op_type +
                AddQuantDequantPass._supported_quantizable_op_type))

        # Check inputs
        assert executor is not None, "The executor cannot be None."
        assert model_dir is not None, "The model_dir cannot be None."
        assert any([gen is not None] for gen in [sample_generator,
            batch_generator]), "The sample_generator and batch_generator " \
            "cannot be None in the same time."
        assert batch_size > 0, "The batch_size should be greater than 0."
        assert algo in self._support_algo_type, \
            "The algo should be KL, abs_max or min_max."
        assert activation_quantize_type in self._support_activation_quantize_type, \
            "The activation_quantize_type ({}) should in ({}).".format(
            activation_quantize_type, self._support_activation_quantize_type)
        assert weight_quantize_type in self._support_weight_quantize_type, \
            "The weight_quantize_type ({}) shoud in ({}).".format(
            weight_quantize_type, self._support_weight_quantize_type)

        # Save input params
        self._executor = executor
        self._scope = global_scope() if scope == None else scope
        self._model_dir = model_dir
        self._model_filename = model_filename
        self._params_filename = params_filename
        self._sample_generator = sample_generator
        self._batch_generator = batch_generator
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._algo = algo
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type
        self._is_full_quantize = is_full_quantize
        if is_full_quantize:
            self._quantizable_op_type = self._support_quantize_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in self._support_quantize_op_type, \
                    op_type + " is not supported for quantization."
        self._optimize_model = optimize_model
        self._is_use_cache_file = is_use_cache_file
        self._cache_dir = cache_dir
        if self._is_use_cache_file and not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        # Define variables
        self._place = self._executor.place
        self._program = None
        self._feed_list = None
        self._fetch_list = None
        self._data_loader = None

        self._out_scale_op_list = _out_scale_op_list
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._sampling_data = {}
        self._quantized_var_kl_threshold = {}
        self._quantized_var_min = {}
        self._quantized_var_max = {}
        self._quantized_var_abs_max = {}

    def quantize(self):
        '''
        Load the FP32 model, and use the calibrate data to calculate the forward-stage.
        Based on the sample data, we can get the quantization information, and obtain
        the final quantized model.

        Args:
            None
        Returns:
            the program of quantized model.
        '''
        self._load_model_data()
        self._collect_target_varnames()
        self._set_activation_persistable()

        batch_id = 0
        for data in self._data_loader():
            self._executor.run(program=self._program,
                               feed=data,
                               fetch_list=self._fetch_list,
                               return_numpy=False,
                               scope=self._scope)
            if self._algo == "KL":
                self._sample_data(batch_id)
            else:
                self._sample_threshold()

            if batch_id % 5 == 0:
                _logger.info("Run batch: " + str(batch_id))
            batch_id += 1
            if self._batch_nums and batch_id >= self._batch_nums:
                break
        _logger.info("Finish all batch: " + str(batch_id))

        self._reset_activation_persistable()

        if self._algo == "KL":
            self._calculate_kl_threshold()

        if self._algo in ["KL", "abs_max"]:
            self._update_program()
        else:
            self._save_input_threhold()

        self._save_output_threshold()
        return self._program

    def save_quantized_model(self,
                             save_model_path,
                             model_filename=None,
                             params_filename=None):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model.
            model_filename(str, optional): If the model_filename is None,
                save the model to '__model__'. Otherwise, save the model
                to the specified filename. Default: None.
            params_filename(str, optional): If the params_filename is None,
                save params to separted files. Otherwise, save all params
                to the specified filename.
        Returns:
            None
        '''
        io.save_inference_model(
            dirname=save_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            feeded_var_names=self._feed_list,
            target_vars=self._fetch_list,
            executor=self._executor,
            main_program=self._program)

    def _load_model_data(self):
        '''
        Load model and set data loader.
        '''
        _logger.info("Load model and set data loader ...")
        [self._program, self._feed_list, self._fetch_list] = \
            io.load_inference_model(dirname=self._model_dir,
                                    executor=self._executor,
                                    model_filename=self._model_filename,
                                    params_filename=self._params_filename)

        if self._program.num_blocks > 1:
            _logger.error("The post training quantization requires that the "
                          "program only has one block.")

        if self._optimize_model:
            self._optimize_fp32_model()

        feed_vars = [framework._get_var(str(var_name), self._program) \
            for var_name in self._feed_list]
        self._data_loader = io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        if self._sample_generator is not None:
            self._data_loader.set_sample_generator(
                self._sample_generator,
                batch_size=self._batch_size,
                drop_last=True,
                places=self._place)
        elif self._batch_generator is not None:
            self._data_loader.set_batch_generator(
                self._batch_generator, places=self._place)

    def _optimize_fp32_model(self):
        '''
        Fuse the `conv2d/depthwise_conv2d + bn` in FP32 model.
        '''
        _logger.info("Optimize FP32 model ...")
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)
        graph = _remove_ctrl_vars(graph)
        graph = _apply_pass(self._scope, graph, 'conv_bn_fuse_pass')
        self._program = graph.to_program()

    def _collect_target_varnames(self):
        '''
        Collect the variable names for sampling, and set activation
        variables to be persistable.
        '''
        # TODO(juncaipeng), consider the name_scope of skip_quant
        _logger.info("Collect quantized variable names ...")

        def collect_var_name(var_name_list, persistable_var_names):
            for var_name in var_name_list:
                if var_name in persistable_var_names:
                    self._quantized_weight_var_name.add(var_name)
                else:
                    self._quantized_act_var_name.add(var_name)

        persistable_var_names = _all_persistable_var_names(self._program)
        for op in self._program.global_block().ops:
            op_type = op.type
            if self._is_full_quantize and \
                op_type not in self._quantizable_op_type:
                _logger.warning(op_type + " is not supported for quantization.")
            # For quantized ops, sample inputs and outputs
            if op_type in self._quantizable_op_type:
                collect_var_name(
                    _get_op_input_var_names(op), persistable_var_names)
                collect_var_name(
                    _get_op_output_var_names(op), persistable_var_names)
            # For other op, only sample output scale
            elif op_type in self._out_scale_op_list:
                collect_var_name(
                    _get_op_output_var_names(op), persistable_var_names)

    def _set_activation_persistable(self):
        '''
        Set activation variables to be persistable, so can obtain 
        the tensor data in sample_data
        '''
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = True

    def _reset_activation_persistable(self):
        '''
        Reset activations to be not persistable.
        '''
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = False

    def _sample_threshold(self):
        '''
        Sample the input threshold(min, max, or abs_max) in every iterations.
        '''
        assert self._algo in ["abs_max", "min_max"], \
            "The algo should be abs_max or min_max to sample min max value."

        if self._algo == "abs_max":
            # Only calculate abs_max value for weight for once
            if self._quantized_var_abs_max == {}:
                for var_name in self._quantized_weight_var_name:
                    var_tensor = _load_variable_data(self._scope, var_name)
                    abs_max_per_channel = []
                    for i in range(var_tensor.shape[0]):
                        abs_max_per_channel.append(
                            float(np.max(np.abs(var_tensor[i]))))
                    self._quantized_var_abs_max[var_name] = abs_max_per_channel
            for var_name in self._quantized_act_var_name:
                var_tensor = _load_variable_data(self._scope, var_name)
                abs_max_value = float(np.max(np.abs(var_tensor)))
                if (var_name not in self._quantized_var_abs_max) or \
                    (abs_max_value > self._quantized_var_abs_max[var_name]):
                    self._quantized_var_abs_max[var_name] = abs_max_value
        elif self._algo == "min_max":
            if self._quantized_var_min == {} and self._quantized_var_max == {}:
                for var_name in self._quantized_weight_var_name:
                    var_tensor = _load_variable_data(self._scope, var_name)
                    min_per_channel = []
                    max_per_channle = []
                    for i in range(var_tensor.shape[0]):
                        min_per_channel.append(float(np.min(var_tensor[i])))
                        max_per_channle.append(float(np.max(var_tensor[i])))
                    self._quantized_var_min[var_name] = min_per_channel
                    self._quantized_var_max[var_name] = max_per_channle
            for var_name in self._quantized_act_var_name:
                var_tensor = _load_variable_data(self._scope, var_name)
                min_value = float(np.min(var_tensor))
                max_value = float(np.max(var_tensor))
                if (var_name not in self._quantized_var_min) or \
                    (min_value < self._quantized_var_min[var_name]):
                    self._quantized_var_min[var_name] = min_value
                if (var_name not in self._quantized_var_max) or \
                    (max_value > self._quantized_var_max[var_name]):
                    self._quantized_var_max[var_name] = max_value

    def _save_input_threhold(self):
        '''
        Save input threshold to the quantized op.
        '''
        assert self._algo == "min_max", \
            "The algo should be min_max to save input threshold."
        for op in self._program.global_block().ops:
            if op.type in self._quantizable_op_type:
                for var_name in _get_op_input_var_names(op):
                    assert var_name in self._quantized_var_min
                    assert var_name in self._quantized_var_max
                    op._set_attr(var_name + ".min",
                                 self._quantized_var_min[var_name])
                    op._set_attr(var_name + ".max",
                                 self._quantized_var_max[var_name])

    def _sample_data(self, iter):
        '''
        Sample the tensor data of quantized variables, 
        applied in every iteration.
        '''
        assert self._algo == "KL", "The algo should be KL to sample data."
        for var_name in self._quantized_weight_var_name:
            if var_name not in self._sampling_data:
                var_tensor = _load_variable_data(self._scope, var_name)
                self._sampling_data[var_name] = var_tensor

        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                save_path = os.path.join(
                    self._cache_dir,
                    var_name.replace("/", ".") + "_" + str(iter) + ".npy")
                np.save(save_path, var_tensor)
        else:
            for var_name in self._quantized_act_var_name:
                if var_name not in self._sampling_data:
                    self._sampling_data[var_name] = []
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                self._sampling_data[var_name].append(var_tensor)

    def _calculate_kl_threshold(self):
        '''
        Calculate the KL threshold of quantized variables.
        '''
        _logger.info("Calculate KL threshold ...")
        assert self._algo == "KL", "The algo should be KL to calculate kl threshold."

        # Abs_max threshold for weights
        for var_name in self._quantized_weight_var_name:
            weight_data = self._sampling_data[var_name]
            weight_threshold = None
            if self._weight_quantize_type == "abs_max":
                weight_threshold = np.max(np.abs(weight_data))
            elif self._weight_quantize_type == "channel_wise_abs_max":
                weight_threshold = []
                for i in range(weight_data.shape[0]):
                    abs_max_value = np.max(np.abs(weight_data[i]))
                    weight_threshold.append(abs_max_value)
            self._quantized_var_kl_threshold[var_name] = weight_threshold

        # KL threshold for activations
        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                sampling_data = []
                filenames = [f for f in os.listdir(self._cache_dir) \
                    if re.match(var_name.replace("/", ".")  + '_[0-9]+.npy', f)]
                for filename in filenames:
                    file_path = os.path.join(self._cache_dir, filename)
                    sampling_data.append(np.load(file_path))
                    os.remove(file_path)
                sampling_data = np.concatenate(sampling_data)
                self._quantized_var_kl_threshold[var_name] = \
                    self._get_kl_scaling_factor(np.abs(sampling_data))
        else:
            for var_name in self._quantized_act_var_name:
                self._sampling_data[var_name] = np.concatenate(
                    self._sampling_data[var_name])
                self._quantized_var_kl_threshold[var_name] = \
                    self._get_kl_scaling_factor(np.abs(self._sampling_data[var_name]))

    def _update_program(self):
        '''
        Use QuantizationTransformPass and AddQuantDequantPass to insert 
        fake_quantize, fake_dequantize and fake_quant_dequant op. 
        Besides, save all kl threshold to the scale var node.
        '''
        _logger.info("Update the program ...")
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)

        # use QuantizationTransformPass to insert fake_quant/fake_dequantize op
        major_quantizable_op_types = []
        for op_type in QuantizationTransformPass._supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                major_quantizable_op_types.append(op_type)
        transform_pass = QuantizationTransformPass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._weight_bits,
            activation_bits=self._activation_bits,
            activation_quantize_type=self._activation_quantize_type,
            weight_quantize_type=self._weight_quantize_type,
            quantizable_op_type=major_quantizable_op_types)
        transform_pass.apply(graph)

        # use AddQuantDequantPass to insert fake_quant_dequant op
        minor_quantizable_op_types = []
        for op_type in AddQuantDequantPass._supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                minor_quantizable_op_types.append(op_type)
        add_quant_dequant_pass = AddQuantDequantPass(
            scope=self._scope,
            place=self._place,
            quantizable_op_type=minor_quantizable_op_types)
        add_quant_dequant_pass.apply(graph)

        # save abs_max or KL threshold to scale var node
        if self._algo == "KL":
            scale_dict = self._quantized_var_kl_threshold
        else:
            scale_dict = self._quantized_var_abs_max
        for key, val in scale_dict.items():
            _set_variable_data(
                self._scope,
                self._place,
                key + ".scale",
                np.array(
                    [val], dtype=np.float32))
            _set_variable_data(
                self._scope,
                self._place,
                key + ".quant_dequant.scale",
                np.array(
                    [val], dtype=np.float32))

        # apply QuantizationFreezePass, and obtain the final quant model
        freeze_pass = QuantizationFreezePass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._weight_bits,
            activation_bits=self._activation_bits,
            weight_quantize_type=self._weight_quantize_type,
            quantizable_op_type=major_quantizable_op_types)
        freeze_pass.apply(graph)
        self._program = graph.to_program()

    def _save_output_threshold(self):
        '''
        Save output threshold to the quantized op.
        '''

        def save_info(op_node, out_var_name, threshold_map, out_info_name,
                      quantized_type):
            assert out_var_name in threshold_map, \
                "The output ({}) of {} node does not have threshold.".format(
                out_var_name, op_node.type)
            op_node._set_attr(out_info_name, threshold_map[var_name])
            if op_node.type in self._quantizable_op_type:
                op._set_attr("quantization_type", quantized_type)

        def analysis_and_save_info(op_node, out_var_name):
            argname_index = _get_output_name_index(op_node, out_var_name)
            assert argname_index is not None, \
                out_var_name + " is not the output of the op"
            if self._algo == "KL":
                # For compatibility, we save output threshold by two methods.
                save_info(op_node, out_var_name,
                          self._quantized_var_kl_threshold, "out_threshold",
                          "post_kl")
                save_info(
                    op_node, out_var_name, self._quantized_var_kl_threshold,
                    argname_index[0] + str(argname_index[1]) + "_threshold",
                    "post_kl")
            elif self._algo == "abs_max":
                save_info(op_node, out_var_name, self._quantized_var_abs_max,
                          "out_threshold", "post_abs_max")
                save_info(
                    op_node, out_var_name, self._quantized_var_abs_max,
                    argname_index[0] + str(argname_index[1]) + "_threshold",
                    "post_kl")
            elif self._algo == "min_max":
                save_info(op_node, out_var_name, self._quantized_var_min,
                          "out_min", "post_min_max")
                save_info(op_node, out_var_name, self._quantized_var_max,
                          "out_max", "post_min_max")

        for op in self._program.global_block().ops:
            if op.type in (self._quantizable_op_type + self._out_scale_op_list):
                out_var_names = _get_op_output_var_names(op)
                assert len(out_var_names) == 1, "Post training " + \
                    "quantization only support one output for " + op.type
                for var_name in out_var_names:
                    analysis_and_save_info(op, var_name)

    def _get_kl_scaling_factor(self, activation_blob, num_quantized_bins=255):
        '''
        Using the KL-divergenc method to get the more precise scaling factor.
        '''
        max_val = np.max(activation_blob)
        min_val = np.min(activation_blob)
        if min_val >= 0:
            hist, hist_edeges = np.histogram(
                activation_blob, bins=2048, range=(min_val, max_val))
            ending_iter = 2047
            starting_iter = int(ending_iter * 0.7)
        else:
            _logger.error("Please first apply abs to activation_blob.")
        bin_width = hist_edeges[1] - hist_edeges[0]

        P_sum = len(np.array(activation_blob).ravel())
        min_kl_divergence = 0
        min_kl_index = 0
        kl_inited = False
        for i in range(starting_iter, ending_iter + 1):
            reference_distr_P = hist[0:i].tolist()
            outliers_count = sum(hist[i:2048])
            if reference_distr_P[i - 1] == 0:
                continue
            reference_distr_P[i - 1] += outliers_count
            reference_distr_bins = reference_distr_P[:]
            candidate_distr_Q = hist[0:i].tolist()
            num_merged_bins = int(i / num_quantized_bins)
            candidate_distr_Q_quantized = [0] * num_quantized_bins
            j_start = 0
            j_end = num_merged_bins
            for idx in range(num_quantized_bins):
                candidate_distr_Q_quantized[idx] = sum(candidate_distr_Q[
                    j_start:j_end])
                j_start += num_merged_bins
                j_end += num_merged_bins
                if (idx + 1) == num_quantized_bins - 1:
                    j_end = i
            candidate_distr_Q = self._expand_quantized_bins(
                candidate_distr_Q_quantized, reference_distr_bins)
            Q_sum = sum(candidate_distr_Q)
            kl_divergence = self._safe_entropy(reference_distr_P, P_sum,
                                               candidate_distr_Q, Q_sum)
            if not kl_inited:
                min_kl_divergence = kl_divergence
                min_kl_index = i
                kl_inited = True
            elif kl_divergence < min_kl_divergence:
                min_kl_divergence = kl_divergence
                min_kl_index = i
            else:
                pass
        if min_kl_index == 0:
            while starting_iter > 0:
                if hist[starting_iter] == 0:
                    starting_iter -= 1
                    continue
                else:
                    break
            min_kl_index = starting_iter
        return (min_kl_index + 0.5) * bin_width

    def _expand_quantized_bins(self, quantized_bins, reference_bins):
        '''
        '''
        expanded_quantized_bins = [0] * len(reference_bins)
        num_merged_bins = int(len(reference_bins) / len(quantized_bins))
        j_start = 0
        j_end = num_merged_bins
        for idx in range(len(quantized_bins)):
            zero_count = reference_bins[j_start:j_end].count(0)
            num_merged_bins = j_end - j_start
            if zero_count == num_merged_bins:
                avg_bin_ele = 0
            else:
                avg_bin_ele = quantized_bins[idx] / (
                    num_merged_bins - zero_count + 0.0)
            for idx1 in range(j_start, j_end):
                expanded_quantized_bins[idx1] = (0 if reference_bins[idx1] == 0
                                                 else avg_bin_ele)
            j_start += num_merged_bins
            j_end += num_merged_bins
            if (idx + 1) == len(quantized_bins) - 1:
                j_end = len(reference_bins)
        return expanded_quantized_bins

    def _safe_entropy(self, reference_distr_P, P_sum, candidate_distr_Q, Q_sum):
        '''
        Calculate the entropy.
        '''
        assert len(reference_distr_P) == len(candidate_distr_Q)
        tmp_sum1 = 0
        tmp_sum2 = 0
        for idx in range(len(reference_distr_P)):
            p_idx = reference_distr_P[idx]
            q_idx = candidate_distr_Q[idx]
            if p_idx == 0:
                tmp_sum1 += 0
                tmp_sum2 += 0
            else:
                if q_idx == 0:
                    _logger.error("Fatal error!, idx = " + str(idx) +
                                  " qindex = 0! p_idx = " + str(p_idx))
                tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
                tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
        return (tmp_sum1 - tmp_sum2) / P_sum


class WeightQuantization(object):
    _supported_quantizable_op_type = ['conv2d', 'depthwise_conv2d', 'mul']
    _supported_weight_quantize_type = ['channel_wise_abs_max', 'abs_max']

    def __init__(self, model_dir, model_filename=None, params_filename=None):
        '''
        This class quantizes the weight of some ops to reduce the size of model
        or improve the perforemace.

        Args:
            model_dir(str): The path of the fp32 model that will be quantized,
                and the model and params files are under the path.
            model_filename(str, optional): The name of file to load the inference
                program. If it is None, the default filename '__model__' will
                be used. Default is 'None'.
            params_filename(str, optional): The name of file to load all parameters.
                When all parameters were saved in a single binary file, set it
                as the real filename. If parameters were saved in separate files,
                set it as 'None'. Default is 'None'.
        '''
        self._model_dir = model_dir
        self._model_filename = model_filename
        self._params_filename = params_filename

    def quantize_weight_to_int(self,
                               save_model_dir,
                               save_model_filename=None,
                               save_params_filename=None,
                               quantizable_op_type=["conv2d", "mul"],
                               weight_bits=8,
                               weight_quantize_type="channel_wise_abs_max",
                               generate_test_model=False,
                               threshold_rate=0.0):
        '''
        In order to reduce the size of model, this api quantizes the weight
        of some ops from float32 to int8/16. In the inference stage, the 
        quantized weight will be dequantized to float32 again.
        
        Args:
            save_model_dir(str): The path to save the quantized model.
            save_model_filename(str, optional): The name of file to 
                save the inference program. If it is None, the default 
                filename '__model__' will be used. Default is 'None'.
            save_params_filename(str, optional): The name of file to 
                save all parameters. If it is None, parameters were 
                saved in separate files. If it is not None, all 
                parameters were saved in a single binary file.
            quantizable_op_type(list[str], optional): The list of ops 
                that will be quantized, and the quantized ops should be
                contained in ["conv2d", "depthwise_conv2d", "mul"]. 
                Default is ["conv2d","mul"].
            weight_bits(int, optional): The bits for the quantized weight, 
                and it should be 8 or 16. Default is 8.
            weight_quantize_type(str, optional): quantization type for weights,
                support 'channel_wise_abs_max' and 'abs_max'. Set it as
                'channel_wise_abs_max', the accuracy performs better.
            generate_test_model(bool, optional): If set generate_test_model 
                as True, it saves a fake quantized model, in which the weights 
                are quantized and dequantized. We can use PaddlePaddle to load 
                the fake quantized model and test the accuracy on GPU or CPU.
            threshold_rate(float, optional): This api uses abs_max methd to 
                quantize the weight from float32 to int8/16, and the abs max 
                value is important for quantization diff. When the abs_max 
                value is far away from the center of the numerical distribution, 
                we can set threshold_rate between 1e-6 and 1e-8, so the abs max 
                value will be optimized. Default is 0.0.
        '''
        for op_type in quantizable_op_type:
            assert op_type in self._supported_quantizable_op_type, \
                "Input error:" + op_type + \
                " is not supported for weight quantization."
        assert weight_bits in [8, 16], \
            "Input error: weight_bits should be 8 or 16."
        assert weight_quantize_type in self._supported_weight_quantize_type, \
            "Input error: weight_quantize_type should in {}".format(
                self._supported_weight_quantize_type)

        quantized_model_dir = os.path.join(save_model_dir, "quantized_model")
        self._quantize_weight_to_int(quantized_model_dir, save_model_filename,
                                     save_params_filename, quantizable_op_type,
                                     weight_bits, weight_quantize_type, False,
                                     threshold_rate)

        if generate_test_model:
            test_model_dir = os.path.join(save_model_dir, "test_model")
            self._quantize_weight_to_int(
                test_model_dir, save_model_filename, save_params_filename,
                quantizable_op_type, weight_bits, weight_quantize_type, True,
                threshold_rate)

    def _quantize_weight_to_int(self, save_model_dir, save_model_filename,
                                save_params_filename, quantizable_op_type,
                                weight_bits, weight_quantize_type, for_test,
                                threshold_rate):
        """
        Generate quantized model or fake quantized model.
        """
        # Load model
        place = core.CPUPlace()
        exe = Executor(place)
        scope = global_scope()
        [program, feed_list, fetch_list] = \
            io.load_inference_model(dirname=self._model_dir,
                                    executor=exe,
                                    model_filename=self._model_filename,
                                    params_filename=self._params_filename)

        quantized_ops = []
        for index in range(program.num_blocks):
            block = program.block(index)
            for op in block.ops:
                if op.type in quantizable_op_type:
                    quantized_ops.append(op)

        # Quantize weights
        persistable_var_names = _all_persistable_var_names(program)
        for op in quantized_ops:
            for var_name in op.input_arg_names:
                if var_name in persistable_var_names:
                    if weight_quantize_type == "abs_max":
                        self._weight_abs_max_quantization(
                            scope, place, weight_bits, threshold_rate, op,
                            var_name, for_test)
                    elif weight_quantize_type == "channel_wise_abs_max":
                        self._weight_channel_wise_abs_max_quantization(
                            scope, place, weight_bits, op, var_name, for_test)

        io.save_inference_model(
            dirname=save_model_dir,
            feeded_var_names=feed_list,
            target_vars=fetch_list,
            executor=exe,
            main_program=program,
            model_filename=save_model_filename,
            params_filename=save_params_filename)

    def _weight_abs_max_quantization(self, scope, place, weight_bits,
                                     threshold_rate, op, var_name, for_test):
        '''
        Use abs_max method to quantize weight.
        '''
        quantize_range = (1 << (weight_bits - 1)) - 1
        save_weight_dtype = np.int8 if weight_bits == 8 else np.int16

        # Get quantized scale and weight data
        weight_data = _load_variable_data(scope, var_name)
        if abs(threshold_rate) < 1e-10:
            threshold_value = np.max(np.abs(weight_data))
        else:
            threshold_value = self._calculate_threshold(\
                weight_data, threshold_rate)
            weight_data[weight_data > threshold_value] = threshold_value
            weight_data[weight_data < -threshold_value] = -threshold_value
        scale = threshold_value / quantize_range
        quantized_weight_data = \
            np.around(weight_data / scale).astype(save_weight_dtype)

        # Set weight data
        if not for_test:
            _set_variable_data(scope, place, var_name, quantized_weight_data)
        else:
            dequantized_weight_data = \
                (quantized_weight_data * scale).astype(np.float32)
            _set_variable_data(scope, place, var_name, dequantized_weight_data)

        # Save info
        op._set_attr('quantization_type', 'post_weight_abs_max')
        op._set_attr('quantize_weight_bits', weight_bits)
        op._set_attr(var_name + "_quant_scale", [scale])  # Save as list

    def _weight_channel_wise_abs_max_quantization(
            self, scope, place, weight_bits, op, var_name, for_test):
        ''' 
        Use channel_wise_abs_max method to quantize weight.
        '''
        quantize_range = (1 << (weight_bits - 1)) - 1
        save_weight_dtype = np.int8 if weight_bits == 8 else np.int16

        # Get quantized scale and weight data
        weight_data = _load_variable_data(scope, var_name)
        if op.type == "mul":
            scales, quantized_weight_data = \
                self._mul_channel_wise_quantization(weight_data,
                    quantize_range, save_weight_dtype)
        elif op.type in ["conv2d", "depthwise_conv2d"]:
            scales, quantized_weight_data = \
                self._conv_channel_wise_quantization(weight_data,
                    quantize_range, save_weight_dtype)
        else:
            _logger.error(op.type + " is not supported by weight quantization")

        # Set weight data
        if not for_test:
            _set_variable_data(scope, place, var_name, quantized_weight_data)
        else:
            if op.type == "mul":
                dequantized_weight_data = \
                    self._mul_channel_wise_dequantization(quantized_weight_data, scales)
            elif op.type in ["conv2d", "depthwise_conv2d"]:
                dequantized_weight_data = \
                    self._conv_channel_wise_dequantization(quantized_weight_data, scales)
            else:
                _logger.error(op.type +
                              " is not supported by weight quantization")
            _set_variable_data(scope, place, var_name, dequantized_weight_data)

        # Save info
        op._set_attr('quantization_type', 'post_weight_channel_wise_abs_max')
        op._set_attr('quantize_weight_bits', weight_bits)
        op._set_attr(var_name + "_quant_scale", scales)

    def _conv_channel_wise_quantization(self, weight_data, quantize_range,
                                        save_weight_dtype):
        '''
        Get channel wise scale for the weights of conv2d and depthwise_conv2d,
        and quantize the weights.
        '''
        scales = []
        quantized_weight_data = np.zeros_like(
            weight_data, dtype=save_weight_dtype)
        channel_num = weight_data.shape[0]
        for i in range(channel_num):
            scale = np.max(np.abs(weight_data[i])) / quantize_range
            scales.append(scale)
            quantized_weight_data[i] = \
                np.around(weight_data[i] / scale).astype(save_weight_dtype)
        return scales, quantized_weight_data

    def _conv_channel_wise_dequantization(self, quantized_weight_data, scales):
        '''
        For conv2d and depthwise_conv2d, dequantize the weights to fp32.
        '''
        dequantized_weight_data = np.zeros_like(
            quantized_weight_data, dtype=np.float32)
        for i in range(len(scales)):
            dequantized_weight_data[i] = \
                (quantized_weight_data[i] * scales[i]).astype(np.float32)
        return dequantized_weight_data

    def _mul_channel_wise_quantization(self, weight_data, quantize_range,
                                       save_weight_dtype):
        '''
        Get channel wise scale for the weights of conv2d and depthwise_conv2d,
        and quantize the weights.
        '''
        scales = []
        quantized_weight_data = np.zeros_like(
            weight_data, dtype=save_weight_dtype)
        channel_num = weight_data.shape[-1]
        for i in range(channel_num):
            scale = np.max(np.abs(weight_data[:, i])) / quantize_range
            scales.append(scale)
            quantized_weight_data[:, i] = \
                np.around(weight_data[:, i] / scale).astype(save_weight_dtype)
        return scales, quantized_weight_data

    def _mul_channel_wise_dequantization(self, quantized_weight_data, scales):
        '''
        For mul, dequantize the weights to fp32.
        '''
        dequantized_weight_data = np.zeros_like(
            quantized_weight_data, dtype=np.float32)
        for i in range(len(scales)):
            dequantized_weight_data[:, i] = \
                (quantized_weight_data[:, i] * scales[i]).astype(np.float32)
        return dequantized_weight_data

    def _calculate_threshold(self, input, threshold_rate, histogram_bins=5000):
        input_abs = np.abs(input)
        hist, hist_edeges = np.histogram(
            input_abs, bins=histogram_bins, range=(0, np.max(input_abs)))
        hist = hist / float(sum(hist))
        hist_sum = 0
        hist_index = 0
        for i in range(len(hist)):
            hist_sum += hist[i]
            if hist_sum >= 1.0 - threshold_rate:
                hist_index = i + 1
                break
        bin_width = hist_edeges[1] - hist_edeges[0]
        return hist_index * bin_width
