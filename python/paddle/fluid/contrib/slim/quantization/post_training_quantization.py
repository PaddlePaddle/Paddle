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
import logging
import numpy as np
from ....executor import global_scope
from .... import io
from .... import core
from .... import framework
from ....framework import IrGraph
from ....log_helper import get_logger
from .quantization_pass import QuantizationTransformPass
from .quantization_pass import QuantizationFreezePass
from .quantization_pass import AddQuantDequantPass

__all__ = ['PostTrainingQuantization']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class PostTrainingQuantization(object):
    def __init__(self,
                 executor,
                 model_path,
                 data_reader,
                 batch_size=10,
                 batch_nums=None,
                 scope=None,
                 algo="KL",
                 quantizable_op_type=[
                     "conv2d", "depthwise_conv2d", "mul", "pool2d",
                     "elementwise_add"
                 ]):
        '''
        The class utilizes post training quantization methon to quantize the 
        fp32 model. It uses calibrate data to calculate the scale factor of 
        quantized variables, and inserts fake quant/dequant op to obtain the 
        quantized model.

        Args:
            executor(fluid.Executor): The executor to load, run and save the 
                quantized model.
            model_path(str): The path of fp32 model that will be quantized.
            data_reader(Reader): The data reader generates a simple every time,
                and it provides calibrate data for DataLoader.
            batch_size(int, optional): The batch size of DataLoader, default is 10.
            batch_nums(int, optional): If set batch_nums, the number of calibrate 
                data is batch_size*batch_nums. If batch_nums=None, use all data
                provided by data_reader as calibrate data.
            scope(fluid.Scope, optional): The scope of the program, use it to load 
                and save variables. If scope=None, get scope by global_scope(). 
            algo(str, optional): If algo=KL, use KL-divergenc method to 
                get the more precise scale factor. If algo='direct', use 
                abs_max methon to get the scale factor. Default is KL.
            quantizable_op_type(list[str], optional): List the type of ops 
                that will be quantized. Default is ["conv2d", "depthwise_conv2d", 
                "mul", "pool2d", "elementwise_add"].
        Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
            
            exe = fluid.Executor(fluid.CPUPlace())
            model_path = path_to_fp32_model
            save_model_path = save_to_
            data_reader =  your_data_reader
            batch_size = 10
            batch_nums = 10
            algo = "KL"
            quantizable_op_type = ["conv2d", \
                "depthwise_conv2d", "mul", "pool2d", "elementwise_add"]
            ptq = PostTrainingQuantization(
                        executor=exe,
                        model_path=model_path,
                        data_reader=data_reader,
                        batch_size=batch_size,
                        batch_nums=batch_nums,
                        algo=algo,
                        quantizable_op_type=quantizable_op_type)
            ptq.quantize_model()
            ptq.save_quantized_model(save_model_path)
        '''
        self._executor = executor
        self._model_path = model_path
        self._data_reader = data_reader
        self._batch_size = batch_size
        self._batch_nums = batch_nums
        self._scope = global_scope() if scope == None else scope
        self._quantizable_op_type = quantizable_op_type
        self._algo = algo
        supported_quantizable_op_type = [
            "conv2d", "depthwise_conv2d", "mul", "pool2d", "elementwise_add"
        ]
        for op_type in self._quantizable_op_type:
            assert op_type in supported_quantizable_op_type, \
                op_type + " is not supported for quantization."

        self._place = self._executor.place
        self._program = None
        self._feed_list = None
        self._fetch_list = None
        self._data_loader = None

        self._bit_length = 8
        self._quantized_weight_var_name = []
        self._quantized_act_var_name = []
        self._sampling_data = {}
        self._quantized_var_scale_factor = {}

    def quantize_model(self):
        '''
        Quantize the fp32 model. Use calibrate data to calculate the scale factor of 
        quantized variables, and inserts fake quant/dequant op to obtain the 
        quantized model.
        
        Return:
            the quantized program.
        '''
        self._prepare()

        batch_id = 0
        for data in self._data_loader():
            self._executor.run(program=self._program,
                               feed=data,
                               fetch_list=self._fetch_list)
            self._sample_data()

            if batch_id % 10 == 0:
                print("train:" + str(batch_id))
            batch_id += 1
            if self._batch_nums and batch_id > self._batch_nums:
                break

        self._calculate_scale_factor()
        self._update_program()

        return self._program

    def save_quantized_model(self, save_model_path):
        '''
        Save the quantized model to the disk.

        Args:
            save_model_path(str): The path to save the quantized model
        Return:
            None
        '''
        io.save_inference_model(
            dirname=save_model_path,
            feeded_var_names=self._feed_list,
            target_vars=self._fetch_list,
            executor=self._executor,
            main_program=self._program)

    def _prepare(self):
        '''
        Load model and set data loader, collect the variable names for sampling, 
        and set activation variables to be persistable.
        '''
        # load model and set data loader
        [self._program, self._feed_list, self._fetch_list] = \
            io.load_inference_model(self._model_path, self._executor)
        feed_vars = [framework._get_var(str(var_name), self._program) \
            for var_name in self._feed_list]
        self._data_loader = io.DataLoader.from_generator(
            feed_list=feed_vars, capacity=3 * self._batch_size, iterable=True)
        self._data_loader.set_sample_generator(
            self._data_reader,
            batch_size=self._batch_size,
            drop_last=False,
            places=self._place)

        #collect the variable names for sampling
        persistable_var_names = []
        for var in self._program.list_vars():
            if var.persistable:
                persistable_var_names.append(var.name)

        block = self._program.global_block()
        for op in block.ops:
            op_type = op.type
            if op_type in self._quantizable_op_type:
                if op_type in ("conv2d", "depthwise_conv2d"):
                    self._quantized_act_var_name.append(op.input("Input")[0])
                    self._quantized_weight_var_name.append(
                        op.input("Filter")[0])
                    self._quantized_act_var_name.append(op.output("Output")[0])
                elif op_type == "mul":
                    x_var_name = op.input("X")[0]
                    y_var_name = op.input("Y")[0]
                    if x_var_name not in persistable_var_names and \
                        y_var_name not in persistable_var_names:
                        op._set_attr("skip_quant", True)
                        _logger.warning("A mul op skip quant for two "
                                        "input variables are not persistable")
                    else:
                        self._quantized_act_var_name.append(x_var_name)
                        self._quantized_weight_var_name.append(y_var_name)
                        self._quantized_act_var_name.append(op.output("Out")[0])
                elif op_type == "pool2d":
                    self._quantized_act_var_name.append(op.input("X")[0])
                elif op_type == "elementwise_add":
                    x_var_name = op.input("X")[0]
                    y_var_name = op.input("Y")[0]
                    if x_var_name not in persistable_var_names and \
                        y_var_name not in persistable_var_names:
                        self._quantized_act_var_name.append(x_var_name)
                        self._quantized_act_var_name.append(y_var_name)

        # set activation variables to be persistable, 
        # so can obtain the tensor data in sample_data stage
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = True

    def _sample_data(self):
        '''
        Sample the tensor data of quantized variables, 
        applied in every iteration.
        '''
        for var_name in self._quantized_weight_var_name:
            if var_name not in self._sampling_data:
                var_tensor = self._load_var_value(var_name)
                self._sampling_data[var_name] = var_tensor

        for var_name in self._quantized_act_var_name:
            if var_name not in self._sampling_data:
                self._sampling_data[var_name] = []
            var_tensor = self._load_var_value(var_name)
            self._sampling_data[var_name].append(var_tensor)

    def _calculate_scale_factor(self):
        '''
        Calculate the scale factor of quantized variables.
        '''
        for var_name in self._quantized_weight_var_name:
            data = self._sampling_data[var_name]
            scale_factor_per_channel = []
            for i in range(data.shape[0]):
                abs_max_value = np.max(np.abs(data[i]))
                scale_factor_per_channel.append(abs_max_value)
            self._quantized_var_scale_factor[
                var_name] = scale_factor_per_channel

        for var_name in self._quantized_act_var_name:
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
        for var in self._program.list_vars():
            if var.name in self._quantized_act_var_name:
                var.persistable = False

        # use QuantizationTransformPass to insert fake_quantize/fake_dequantize op
        graph = IrGraph(core.Graph(self._program.desc), for_test=True)

        qtp_quantizable_op_type = []
        for op_type in ["conv2d", "depthwise_conv2d", "mul"]:
            if op_type in self._quantizable_op_type:
                qtp_quantizable_op_type.append(op_type)
        transform_pass = QuantizationTransformPass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._bit_length,
            activation_bits=self._bit_length,
            activation_quantize_type='moving_average_abs_max',
            weight_quantize_type='channel_wise_abs_max',
            quantizable_op_type=qtp_quantizable_op_type)
        transform_pass.apply(graph)

        # use AddQuantDequantPass to insert fake_quant_dequant op
        aqdp_quantizable_op_type = []
        for op_type in ["pool2d", "elementwise_add"]:
            if op_type in self._quantizable_op_type:
                aqdp_quantizable_op_type.append(op_type)
        add_quant_dequant_pass = AddQuantDequantPass(
            scope=self._scope,
            place=self._place,
            quantizable_op_type=aqdp_quantizable_op_type)
        add_quant_dequant_pass.apply(graph)

        # save scale factor to scale var node
        for key, val in self._quantized_var_scale_factor.items():
            self._set_var_node_value(
                key + ".scale", np.array(
                    [val], dtype=np.float32))
            self._set_var_node_value(
                key + ".quant_dequant.scale", np.array(
                    [val], dtype=np.float32))

        # apply QuantizationFreezePass, and obtain the final quant model
        freeze_pass = QuantizationFreezePass(
            scope=self._scope,
            place=self._place,
            weight_bits=self._bit_length,
            activation_bits=self._bit_length,
            weight_quantize_type='channel_wise_abs_max',
            quantizable_op_type=qtp_quantizable_op_type)
        freeze_pass.apply(graph)
        self._program = graph.to_program()

    def _load_var_value(self, var_name):
        '''
        Load variable value from scope
        '''
        return np.array(self._scope.find_var(var_name).get_tensor())

    def _set_var_node_value(self, var_node_name, np_value):
        '''
        Set the value of var node by name, if the node is not exits,
        '''
        assert isinstance(np_value, np.ndarray), \
            'The type of value should be numpy array.'
        var_node = self._scope.find_var(var_node_name)
        if var_node != None:
            tensor = var_node.get_tensor()
            tensor.set(np_value, self._place)

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
                    print("Fatal error!, idx = " + str(idx) +
                          " qindex = 0! p_idx = " + str(p_idx))
                tmp_sum1 += p_idx * (math.log(Q_sum * p_idx))
                tmp_sum2 += p_idx * (math.log(P_sum * q_idx))
        return (tmp_sum1 - tmp_sum2) / P_sum
