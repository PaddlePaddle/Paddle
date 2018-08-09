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
"""Merge parameters value and model config into a standalone file.
This script is designed to take a model config(.py) and the parameters
values stored in a gzip file, and output a sigle file.

In addition to the above function, we also offer two optional useful tools
during Merge process:

1. rounding
Given floating parameters V, first we quantify the float32 V to 8-bit
integers V'. And then we transformed back V' back into its approximate
high-precision value by performing the inverse of the quantization operation.
We then perform gzip to the parameters. This process can reduces the 
parameter size by 70%

2. merge_batch_normazlization
The batch normalization followed the convolution or fully connected layer can be 
integrated with them. Doing so will give us a forward acceleration(about 30%
in mobilenet).

It's useful to do this when we need to load a single file in C++, especially in
environments like mobile or embedded. 
"""

import gzip
import struct
import os
import numpy as np
import math
import copy

from paddle.trainer_config_helpers.layers import LayerOutput
from paddle.v2.parameters import Parameters
from paddle.proto import ModelConfig_pb2
from paddle.v2.topology import Topology


def merge_v2_model(net,
                   param_file,
                   output_file,
                   with_rounding=False,
                   merge_batch_normazlization=False):
    '''Merge the model config and parameters into one file.

    The model configuration file describes the model structure which
    ends with .py. The parameters file stores the parameters of the model
    which ends with .tar.gz.

    @param  net            The output layer of the network for inference.
    @param  param_file     Path of the parameters (.tar.gz) which is stored by
                           v2 api.
    @param  output_file    Path of the merged file which will be generated.
    @param  with_rounding  Whether to round weight
    @param  merge_batch_normazlization  Whether merge batch normalization.

    Usage:

        from paddle.utils.merge_model import merge_v2_model
        # import your network configuration
        from example_net import net_conf

        net = net_conf(is_predict=True)
        param_file = './param_pass_00000.tar.gz'
        output_file = './output.paddle'

        merge_v2_model(net,
                       param_file,
                       output_file,
                       with_rounding = False,
                       merge_batch_normazlization = True)

    '''

    assert isinstance(net, LayerOutput), \
            "The net should be the output of the network for inference"
    assert os.path.exists(param_file), \
            "The model parameters file %s does not exists " % (param_file)

    model_proto = Topology(net).proto()
    assert isinstance(model_proto, ModelConfig_pb2.ModelConfig)

    with gzip.open(param_file) as f:
        params = Parameters.from_tar(f)

    if with_rounding:
        rd = Round(params)
        params = rd.do_round()

    if merge_batch_normazlization:
        mb = Merge_BN(model_proto, params)
        model_proto, params = mb.merge()

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f:
        param_names = [param.name for param in model_proto.parameters]
        conf_str = model_proto.SerializeToString()
        f.write(struct.pack('q', len(conf_str)))
        f.write(conf_str)
        for pname in param_names:
            params.serialize(pname, f)

    print 'Generate  %s  success!' % (output_file)


class Round:
    def __init__(self, source_param, scale_section=(2**8) - 1):
        self.param = source_param
        self.scale_section = scale_section

    def do_round(self):
        all_params = self.param.keys()
        for param_name in all_params:
            param_value = self.param[param_name]
            param_shape = param_value.shape
            param_size = param_value.size
            if param_size > 8192:
                print('Rounding ...... ' + param_name + ' size: ' +
                      str(param_size))
                param_value = param_value.reshape(-1)
                param_min = np.min(param_value)
                param_max = np.max(param_value)
                bucket_width = (param_max - param_min) / self.scale_section

                for index in range(param_size):
                    if (param_value[index] != 0):
                        bucket = math.floor(
                            (param_value[index] - param_min) / bucket_width)
                        param_value[index] = np.float32((
                            bucket + 0.5) * bucket_width + param_min)
                param_value = param_value.reshape(param_shape)
            else:
                # print ("We retain the original %s because it's size is below 8192" % param_name)
                pass
        return self.param


class Merge_BN:
    def __init__(self, source_proto, source_param):
        self.source_proto = source_proto
        self.parameters = self.source_proto.parameters
        self.param_names = [param.name for param in self.parameters]

        self.layers = self.source_proto.layers
        self.sub_models = self.source_proto.sub_models[0]
        self.input_layers_name = {}
        self.delete_param_name = []
        self.delete_layer_name = []

        self.source_param = source_param

    def fuse_param(self, current_layer, bn_layer):
        current_layer.active_type = bn_layer.active_type

        param_name = current_layer.inputs[0].input_parameter_name
        bias_name = current_layer.bias_parameter_name
        assert param_name, 'This layer(fc or exconv) should have parameters'

        bn_inputs = bn_layer.inputs

        a_bn_name = bn_inputs[0].input_parameter_name
        mean_bn_name = bn_inputs[1].input_parameter_name
        var_bn_name = bn_inputs[2].input_parameter_name
        b_bn_name = bn_layer.bias_parameter_name

        a_bn = self.source_param.get(a_bn_name)
        mean_bn = self.source_param.get(mean_bn_name)
        var_bn = self.source_param.get(var_bn_name)
        b_bn = self.source_param.get(b_bn_name)

        param = self.source_param.get(param_name)
        param_shape = param.shape

        bias = np.zeros(a_bn.shape[1])
        bias_shape = (bias.size, 1)

        # the parameters in batch normalization will be unuseful
        self.delete_param_name.append(a_bn_name)
        self.delete_param_name.append(mean_bn_name)
        self.delete_param_name.append(var_bn_name)

        if not bias_name:
            bias_name = param_name.split('.')[0] + '.wbias'
            current_layer.bias_parameter_name = bias_name
            # 1.add conv bias parameter config to prototxt
            # 2.add conv bias parameters config to prameters
            bias_shape = b_bn.shape
            bias_param_config = self.parameters[self.param_names.index(
                b_bn_name)]
            bias_param_config.name = bias_name
            self.source_param.__append_config__(bias_param_config)
        else:
            self.delete_param_name.append(b_bn_name)
            bias = self.source_param.get(bias_name).reshape(1, -1)

        std_bn = np.float32(np.sqrt(np.add(var_bn, 1e-5)))
        tmp1 = np.float32(np.divide(a_bn, std_bn))
        bias = np.float32(
            np.add(np.multiply(np.subtract(bias, mean_bn), tmp1),
                   b_bn)).reshape(bias_shape)
        self.source_param.set(bias_name, bias)
        tmp1 = tmp1.reshape(tmp1.shape[1], -1)
        param = param.reshape((tmp1.shape[0], -1))
        param = np.float32(np.multiply(param, tmp1)).reshape(param_shape)
        self.source_param.set(param_name, param)

    def delete_useless_param(self):
        temp_parameters = copy.deepcopy(self.parameters)
        for param in temp_parameters:
            if param.name in self.delete_param_name:
                self.parameters.remove(param)

    def adjust_input(self):
        cut_layer_names = self.input_layers_name.keys()
        for layer in self.layers:
            for input in layer.inputs:
                input_name = input.input_layer_name
                if input_name in cut_layer_names:
                    input.input_layer_name = self.input_layers_name[input_name]

        # Deal with the situation if output layer is batch normalization.
        for i, output_layer_name in enumerate(
                self.source_proto.output_layer_names):
            if output_layer_name in cut_layer_names:
                self.source_proto.output_layer_names[
                    i] = self.input_layers_name[output_layer_name]

        for i, output_layer_name in enumerate(
                self.sub_models.output_layer_names):
            if output_layer_name in cut_layer_names:
                self.sub_models.output_layer_names[i] = self.input_layers_name[
                    output_layer_name]

    def delete_useless_layer(self):
        temp_layers = copy.deepcopy(self.layers)
        temp_layer_names = copy.deepcopy(self.sub_models.layer_names)
        for layer in temp_layers:
            if layer.name in self.delete_layer_name:
                self.layers.remove(layer)

        for layer_name in temp_layer_names:
            if layer_name in self.delete_layer_name:
                self.sub_models.layer_names.remove(layer_name)

    def merge(self):
        layer_num = len(self.layers)
        i = 0
        while i < layer_num:
            current_layer = self.layers[i]
            if current_layer.type in ['exconv', 'fc', 'exconvt']:
                if (i + 1 < layer_num and
                        self.layers[i + 1].type == 'batch_norm'):
                    self.fuse_param(current_layer, self.layers[i + 1])
                    self.input_layers_name[self.layers[i + 1]
                                           .name] = current_layer.name
                    self.delete_layer_name.append(self.layers[i + 1].name)
                    i += 2
                    continue
            i = i + 1

        self.delete_useless_param()
        self.delete_useless_layer()
        self.adjust_input()
        return self.source_proto, self.source_param
