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
from .quantization_pass import _op_real_in_out_name

__all__ = ['PostTrainingQuantization', 'WeightQuantization']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    return np.array(scope.find_var(var_name).get_tensor())


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


class PostTrainingQuantization(object):
    def __init__(self,
                 executor=None,
                 scope=None,
                 model_dir=None,
                 model_filename=None,
                 params_filename=None,
                 sample_generator=None,
                 batch_size=10,
                 batch_nums=None,
                 algo="KL",
                 quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
                 is_full_quantize=False,
                 weight_bits=8,
                 activation_bits=8,
                 is_use_cache_file=False,
                 cache_dir="./temp_post_training"):
        '''
        The class utilizes post training quantization methon to quantize the 
        fp32 model. It uses calibrate data to calculate the scale factor of 
        quantized variables, and inserts fake quant/dequant op to obtain the 
        quantized model.

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
            sample_generator(Python Generator): The sample generator provides 
                calibrate data for DataLoader, and it only returns a sample every 
                time.
            batch_size(int, optional): The batch size of DataLoader. Default is 10.
            batch_nums(int, optional): If batch_nums is not None, the number of 
                calibrate data is batch_size*batch_nums. If batch_nums is None, use 
                all data provided by sample_generator as calibrate data.
            algo(str, optional): If algo=KL, use KL-divergenc method to 
                get the more precise scale factor. If algo='direct', use 
                abs_max methon to get the scale factor. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops 
                that will be quantized. Default is ["conv2d", "depthwise_conv2d", 
                "mul"].
            is_full_quantized(bool, optional): If set is_full_quantized as True, 
                apply quantization to all supported quantizable op type. If set
                is_full_quantized as False, only apply quantization to the op type 
                according to the input quantizable_op_type.
            weight_bits(int, optional): quantization bit number for weights.
            activation_bits(int): quantization bit number for activation.
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

        assert executor is not None, "The executor cannot be None."
        assert model_dir is not None, "The model_dir cannot be None."
        assert sample_generator is not None, \
                "The sample_generator cannot be None."

        self._executor = executor
        self._scope = global_scope() if scope == None else scope
        self._model_dir = model_dir
        self._model_filename = model_filename
        self._params_filename = params_filename
        self._sample_generator = sample_generator
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._algo = algo
        self._is_use_cache_file = is_use_cache_file
        self._cache_dir = cache_dir
        if self._is_use_cache_file and not os.path.exists(self._cache_dir):
            os.mkdir(self._cache_dir)

        supported_quantizable_op_type = \
            QuantizationTransformPass._supported_quantizable_op_type + \
            AddQuantDequantPass._supported_quantizable_op_type
        if is_full_quantize:
            self._quantizable_op_type = supported_quantizable_op_type
        else:
            self._quantizable_op_type = quantizable_op_type
            for op_type in self._quantizable_op_type:
                assert op_type in supported_quantizable_op_type + \
                    AddQuantDequantPass._activation_type, \
                    op_type + " is not supported for quantization."

        self._place = self._executor.place
        self._program = None
        self._feed_list = None
        self._fetch_list = None
        self._data_loader = None

        self._op_real_in_out_name = _op_real_in_out_name
        self._bit_length = 8
        self._quantized_weight_var_name = set()
        self._quantized_act_var_name = set()
        self._sampling_data = {}
        self._quantized_var_scale_factor = {}

    def quantize(self):
        '''
        Quantize the fp32 model. Use calibrate data to calculate the scale factor of 
        quantized variables, and inserts fake quant/dequant op to obtain the 
        quantized model.

        Args:
            None
        Returns:
            the program of quantized model.
        '''
        self._preprocess()

        batch_id = 0
        for data in self._data_loader():
            self._executor.run(program=self._program,
                               feed=data,
                               fetch_list=self._fetch_list,
                               return_numpy=False)
            self._sample_data(batch_id)

            if batch_id % 5 == 0:
                _logger.info("run batch: " + str(batch_id))
            batch_id += 1
            if self._batch_nums and batch_id >= self._batch_nums:
                break
        _logger.info("all run batch: " + str(batch_id))

        _logger.info("calculate scale factor ...")
        self._calculate_scale_factor()

        _logger.info("update the program ...")
        self._update_program()

        self._save_output_scale()
        return self._program

    def save_quantized_model(self, save_model_path):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model
        Returns:
            None
        '''
        io.save_inference_model(
            dirname=save_model_path,
            feeded_var_names=self._feed_list,
            target_vars=self._fetch_list,
            executor=self._executor,
            main_program=self._program)

    def _preprocess(self):
        '''
        Load model and set data loader, collect the variable names for sampling, 
        and set activation variables to be persistable.
        '''
        # load model and set data loader
        [self._program, self._feed_list, self._fetch_list] = \
            io.load_inference_model(dirname=self._model_dir,
                                    executor=self._executor,
                                    model_filename=self._model_filename,
                                    params_filename=self._params_filename)
        feed_vars = [framework._get_var(str(var_name), self._program) \
            for var_name in self._feed_list]
        self._data_loader = io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        self._data_loader.set_sample_generator(
            self._sample_generator,
            batch_size=self._batch_size,
            drop_last=True,
            places=self._place)

        # collect the variable names for sampling.
        # TODO(juncaipeng), consider the name_scope of skip_quant and
        # reduce the variables for sampling
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        for op in self._program.global_block().ops:
            op_type = op.type
            if op_type in self._quantizable_op_type:
                if op_type in ("conv2d", "depthwise_conv2d"):
                    self._quantized_act_var_name.add(op.input("Input")[0])
                    self._quantized_weight_var_name.add(op.input("Filter")[0])
                    self._quantized_act_var_name.add(op.output("Output")[0])
                elif op_type in ["mul", "matmul"]:
                    x_var_name = op.input("X")[0]
                    if x_var_name in persistable_var_names:
                        self._quantized_weight_var_name.add(x_var_name)
                    else:
                        self._quantized_act_var_name.add(x_var_name)
                    y_var_name = op.input("Y")[0]
                    if y_var_name in persistable_var_names:
                        self._quantized_weight_var_name.add(y_var_name)
                    else:
                        self._quantized_act_var_name.add(y_var_name)
                    self._quantized_act_var_name.add(op.output("Out")[0])
                else:
                    # process other quantizable op type, the input must all not persistable
                    if self._is_input_all_not_persistable(
                            op, persistable_var_names):
                        input_output_name_list = self._op_real_in_out_name[
                            op_type]
                        for input_name in input_output_name_list[0]:
                            for var_name in op.input(input_name):
                                self._quantized_act_var_name.add(var_name)
                        for output_name in input_output_name_list[1]:
                            for var_name in op.output(output_name):
                                self._quantized_act_var_name.add(var_name)

        # set activation variables to be persistable, so can obtain 
        # the tensor data in sample_data
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = True

    def _sample_data(self, iter):
        '''
        Sample the tensor data of quantized variables, 
        applied in every iteration.
        '''
        for var_name in self._quantized_weight_var_name:
            if var_name not in self._sampling_data:
                var_tensor = _load_variable_data(self._scope, var_name)
                self._sampling_data[var_name] = var_tensor

        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                save_path = os.path.join(self._cache_dir,
                                         var_name + "_" + str(iter) + ".npy")
                np.save(save_path, var_tensor)
        else:
            for var_name in self._quantized_act_var_name:
                if var_name not in self._sampling_data:
                    self._sampling_data[var_name] = []
                var_tensor = _load_variable_data(self._scope, var_name)
                var_tensor = var_tensor.ravel()
                self._sampling_data[var_name].append(var_tensor)

    def _calculate_scale_factor(self):
        '''
        Calculate the scale factor of quantized variables.
        '''
        # apply channel_wise_abs_max quantization for weights
        for var_name in self._quantized_weight_var_name:
            data = self._sampling_data[var_name]
            scale_factor_per_channel = []
            for i in range(data.shape[0]):
                abs_max_value = np.max(np.abs(data[i]))
                scale_factor_per_channel.append(abs_max_value)
            self._quantized_var_scale_factor[
                var_name] = scale_factor_per_channel

        # apply kl quantization for activation
        if self._is_use_cache_file:
            for var_name in self._quantized_act_var_name:
                sampling_data = []
                filenames = [f for f in os.listdir(self._cache_dir) \
                    if re.match(var_name + '_[0-9]+.npy', f)]
                for filename in filenames:
                    file_path = os.path.join(self._cache_dir, filename)
                    sampling_data.append(np.load(file_path))
                    os.remove(file_path)
                sampling_data = np.concatenate(sampling_data)

                if self._algo == "KL":
                    self._quantized_var_scale_factor[var_name] = \
                        self._get_kl_scaling_factor(np.abs(sampling_data))
                else:
                    self._quantized_var_scale_factor[var_name] = \
                        np.max(np.abs(sampling_data))
        else:
            for var_name in self._quantized_act_var_name:
                self._sampling_data[var_name] = np.concatenate(
                    self._sampling_data[var_name])
                if self._algo == "KL":
                    self._quantized_var_scale_factor[var_name] = \
                        self._get_kl_scaling_factor(np.abs(self._sampling_data[var_name]))
                else:
                    self._quantized_var_scale_factor[var_name] = \
                        np.max(np.abs(self._sampling_data[var_name]))

    def _update_program(self):
        '''
        Insert fake_quantize/fake_dequantize op to the program.
        '''
        # reset quantized activation variable
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = False

        # use QuantizationTransformPass to insert fake_quantize/fake_dequantize op
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)

        major_quantizable_op_types = []
        for op_type in QuantizationTransformPass._supported_quantizable_op_type:
            if op_type in self._quantizable_op_type:
                major_quantizable_op_types.append(op_type)
        transform_pass = QuantizationTransformPass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._bit_length,
            activation_bits=self._bit_length,
            activation_quantize_type='moving_average_abs_max',
            weight_quantize_type='channel_wise_abs_max',
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

        # save scale factor to scale var node
        for key, val in self._quantized_var_scale_factor.items():
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
            weight_bits=self._bit_length,
            activation_bits=self._bit_length,
            weight_quantize_type='channel_wise_abs_max',
            quantizable_op_type=major_quantizable_op_types)
        freeze_pass.apply(graph)
        self._program = graph.to_program()

    def _save_output_scale(self):
        '''
        Save output scale to the quantized op.
        '''
        output_scale_name = "output_scale"
        for op in self._program.global_block().ops:
            if op.type in self._quantizable_op_type:
                output_name_list = self._op_real_in_out_name[op.type][1]
                for output_name in output_name_list:
                    for output_var_name in op.output(output_name):
                        if output_var_name in self._quantized_var_scale_factor:
                            op._set_attr(output_scale_name,
                                         self._quantized_var_scale_factor[
                                             output_var_name])

    def _is_input_all_not_persistable(self, op, persistable_var_names):
        '''
        Analyze the real inputs of the op are all not persistable.
        '''
        is_input_all_not_persistable = True
        input_name_list = self._op_real_in_out_name[op.type][0]
        for input_name in input_name_list:
            for var_name in op.input(input_name):
                if var_name in persistable_var_names:
                    is_input_all_not_persistable = False
                    break
        return is_input_all_not_persistable

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
            threshold_rate(float, optional): This api uses abs_max methd to 
                quantize the weight from float32 to int8/16, and the abs max 
                value is important for quantization diff. When the abs_max 
                value is far away from the center of the numerical distribution, 
                we can set threshold_rate between 1e-6 and 1e-8, so the abs max 
                value will be optimized. Default is 0.0.
        '''
        for op_type in quantizable_op_type:
            assert op_type in self._supported_quantizable_op_type, \
                "input error:" + op_type + \
                " is not supported for weight quantization."
        assert weight_bits in [8, 16], \
            "input error: weight_bits should be 8 or 16."
        quantize_range = (1 << (weight_bits - 1)) - 1
        save_weight_dtype = np.int8 if weight_bits == 8 else np.int16

        place = core.CPUPlace()
        exe = Executor(place)
        scope = global_scope()
        [program, feed_list, fetch_list] = \
            io.load_inference_model(dirname=self._model_dir,
                                    executor=exe,
                                    model_filename=self._model_filename,
                                    params_filename=self._params_filename)

        persistable_var_names = []
        for var in program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)
        for op in program.global_block().ops:
            if op.type in quantizable_op_type:
                for var_name in op.input_arg_names:
                    if var_name in persistable_var_names:
                        var_tensor_data = _load_variable_data(scope, var_name)
                        if abs(threshold_rate) < 1e-10:
                            threshold_value = np.max(np.abs(var_tensor_data))
                        else:
                            threshold_value = self._calculate_threshold(\
                                var_tensor_data, threshold_rate)
                            var_tensor_data[var_tensor_data >
                                            threshold_value] = threshold_value
                            var_tensor_data[var_tensor_data <
                                            -threshold_value] = -threshold_value
                        scale = threshold_value / quantize_range
                        quantized_var_tensor_data = \
                            np.around(var_tensor_data / scale)
                        quantized_var_tensor_data = \
                            quantized_var_tensor_data.astype(save_weight_dtype)
                        _set_variable_data(scope, place, var_name,
                                           quantized_var_tensor_data)
                        op._set_attr(var_name + "_quant_scale", [scale])
                        op._set_attr('quantize_weight_bits', weight_bits)

        io.save_inference_model(
            dirname=save_model_dir,
            feeded_var_names=feed_list,
            target_vars=fetch_list,
            executor=exe,
            main_program=program,
            model_filename=save_model_filename,
            params_filename=save_params_filename)

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
