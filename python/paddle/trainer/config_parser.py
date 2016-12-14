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

from __future__ import print_function
'''
The following functions are available in the config file:

Bias: define bias. To be used as value of bias argument in Layer().

Data: define data provider.

Input: define input layer for a layer. To be used as element of inputs argument
       in Layer().

Conv: define a convolution operation for an input of a layer.

Norm: define a normalization operation for an input of a layer.

Pool: define a pooling operation for an input of a layer.

Layer: define a layer.

Parameter: define a parameter.

Import: import another config file. If the imported config file name is
        a relative path, then it will be searched under the directory of the
        current config file.

Inputs(layer_names...):
    Define the name of the input layers of the NeuralNetwork.
    The type of these layers must be "data".
    These layers will be provided with the DataBatch obtained
    from DataProvider. The data streams from DataProvider must
    have the same order.

Outputs(layer_names...):
    Define the name of the output layers of the NeuralNetwork.
    Usually the output is simply the cost layer.
    You can specify other layers as outputs and  calculate the
    cost (and its derivative) yourself.


default_initial_std(val)
default_initial_mean(val)
default_momentum(val):
default_decay_rate(val): Set the default value for these parameters


get_config_arg(name, type, default): Get the value for a config parameter.


*** customized extension to config_parser ***
The functionality of the config_parser can be extended.
If the config_arg_str for parse_config() contains
extension_module_name=[MODULE_NAME], then config_parser will call
MODULE_NAME.get_config_funcs(g_config)
MODULE_NAME.get_config_funcs() should return a dictionary of name to functions,
those functions will be available in the config file.
See trainer/tests/config_parser_test.py for example

To use this from paddle_trainer, paddle_trainer should be called with
--config_args=extension_module_name=[MODULE_NAME]

'''

import copy
import logging
import os
import sys
import traceback
import math
import shutil

try:
    from paddle.proto.DataConfig_pb2 import DataConfig
    from paddle.proto.ModelConfig_pb2 import ModelConfig
    from paddle.proto.ModelConfig_pb2 import LayerConfig
    from paddle.proto.ModelConfig_pb2 import LayerInputConfig
    from paddle.proto.ModelConfig_pb2 import ProjectionConfig
    from paddle.proto.ModelConfig_pb2 import OperatorConfig
    from paddle.proto.ModelConfig_pb2 import GeneratorConfig
    from paddle.proto.ModelConfig_pb2 import LinkConfig
    from paddle.proto.ParameterConfig_pb2 import ParameterConfig
    from paddle.proto.ParameterConfig_pb2 import ParameterUpdaterHookConfig
    from paddle.proto.TrainerConfig_pb2 import TrainerConfig

except Exception as e:
    traceback.print_exc()
    raise

logging.basicConfig(
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s', )
logger = logging.getLogger('paddle')
logger.setLevel(logging.INFO)
__real_print__ = print
print = logger.info

# from layer type name to layer class
g_layer_type_map = {}


# Initialize global variables. We use this function so that we can
# call parse_config() multiple times
def init_config_environment(
        g_default_momentum=None,
        g_default_decay_rate=None,
        g_default_initial_mean=0.,
        g_default_initial_std=0.01,
        g_default_num_batches_regularization=None,
        g_default_initial_strategy=0,
        g_default_initial_smart=False,
        g_default_gradient_clipping_threshold=None,
        g_default_device=None,
        g_default_update_hooks=None,
        g_default_compact_func=None,
        g_config=TrainerConfig(),
        g_layer_map={},
        g_parameter_map={},
        g_extended_config_funcs={},

        # store command args of paddle_trainer
        g_command_config_args={},

        # Used for PyDataProvider to avoid duplicate module name
        g_py_module_name_list=[],
        g_current_submodel=None,
        g_root_submodel=None,
        g_submodel_map={},
        g_submodel_stack=[],
        g_add_submodel_suffix=False,

        # Whether current layer needs to pass the image height and width.
        # Default value is true, but if it encounters recurrent_layer_group,
        # it will be false. The reason is that image is converted to be sequence,
        # image height will be sequence length, and image width will be feature
        # length of each timestep.
        g_pass_height_width=True, ):

    for k, v in locals().iteritems():
        globals()[k] = copy.deepcopy(v)


# Because type is widely used as a variable name in this code.
# we need a different function name for the builtin type()
def type_of(x):
    return type(x)


# Check a condition derived config file
def config_assert(b, msg):
    if not b:
        logger.fatal(msg)


g_config_funcs = {}


# decorator for indicating a function which can be used in config file
def config_func(func):
    g_config_funcs[func.func_name] = func
    return func


# decorator for indicating a class which can be used in config file
def config_class(cls):
    g_config_funcs[cls.__name__] = cls
    return cls


# decorator for indicating a class for a layer type
def config_layer(layer_type):
    def wrap(cls):
        g_config_funcs[cls.__name__] = cls
        g_layer_type_map[layer_type] = cls
        return cls

    return wrap


def gen_parameter_name(layer_name, input_index):
    return '_%s.w%d' % (layer_name, input_index)


def gen_bias_parameter_name(layer_name):
    return '_%s.wbias' % layer_name


def default(x, default_value):
    return default_value if x is None else x


class Cfg(object):
    def add_keys(self, locals):
        for k, v in locals.iteritems():
            if not k.startswith('_'):
                self.__setattr__(k, v)


# functions available in config file


# Define the name of the input layers of the NeuralNetwork.
# The type of these layers must be "data".
# These layers will be provided with the DataBatch obtained
# from DataProvider. The data streams from DataProvider must
# have the same order.
@config_func
def Inputs(*args):
    for name in args:
        name = MakeLayerNameInSubmodel(name)
        global g_current_submodel, g_root_submodel
        if g_current_submodel.is_recurrent_layer_group:
            config_assert(False, "Do not set Inputs in recurrent layer group")
        else:
            g_current_submodel.input_layer_names.append(name)

        if g_current_submodel is g_root_submodel:
            g_config.model_config.input_layer_names.append(name)


@config_func
def HasInputsSet():
    return len(g_current_submodel.input_layer_names) != 0


# Define the name of the output layers of the NeuralNetwork.
# Usually the output is simply the cost layer.
# You can specify other layers as outputs and calculate the
# cost (and its derivative) yourself.
@config_func
def Outputs(*args):
    for name in args:
        name = MakeLayerNameInSubmodel(name)
        global g_current_submodel, g_root_submodel
        if g_current_submodel.is_recurrent_layer_group:
            config_assert(False, "Do not set Outputs in recurrent layer group")
        else:
            g_current_submodel.output_layer_names.append(name)

        if g_current_submodel is g_root_submodel:
            g_config.model_config.output_layer_names.append(name)


@config_func
def SubModelBegin(name):
    global g_current_submodel, g_root_submodel, g_submodel_stack
    g_submodel_stack.append(g_current_submodel)

    name = MakeLayerNameInParentSubmodel(name)  #rename in nested submodel

    config_assert(name not in g_submodel_map,
                  'Duplicated submodel name: %s' % name)

    sub_model = g_config.model_config.sub_models.add()
    sub_model.name = name
    g_submodel_map[name] = sub_model
    g_current_submodel = sub_model


@config_func
def SubModelEnd(name=None):
    global g_current_submodel, g_root_submodel, g_submodel_stack
    config_assert(g_current_submodel is not g_root_submodel,
                  "submodel not begin")
    if name is not None:
        config_assert(
            g_current_submodel.name == MakeLayerNameInParentSubmodel(name),
            "submodel name error")

    g_current_submodel = g_submodel_stack.pop()


def MakeLayerNameInParentSubmodel(name):
    suffix = ""
    if len(g_submodel_stack) > 1:
        suffix = "@" + g_submodel_stack[-1].name
    return name + suffix


def GetLayerBaseName(name):
    return name.split('@')[0]


def MakeLayerNameInSubmodel(name, submodel_name=None):
    global g_current_submodel
    global g_add_submodel_suffix
    if (submodel_name is None and not g_add_submodel_suffix and
            not g_current_submodel.is_recurrent_layer_group):
        return name
    if submodel_name is None:
        submodel_name = g_current_submodel.name
    return name + "@" + submodel_name


# Define a recurrent layer group begin with RecurrentLayerGroupBegin
# and end with RecurrentLayerGroupEnd.
# A recurrent layer group forward/backward one frame after previous frame
# forward/backward through all layers in layer group.
# in_links are names of layer used as input layer in the layer group.
# out_links are names of layer in layer group used as outside layer's input.
#
# If generator is set, the layer group need one or more than one outlinks.
# The first outlink should always be the generated token ids.
# If generator.num_results_per_sample is not set, the output for one sample is
# a ids sequence. Else if num_results_per_sample is more than one,
# the output for one sample is up to #num_results_per_sample generated
# sequences, which are packed in one sequence in output ids vector. Each
# generated sequence has a generation probability. The probabilities for one
# sample are stored in one row of output value matrix.
# Packed generated sequences format, for each i:
#   seq_i_length: one interger, seq_i content length,
#   [seq_i content], length = seq_i_length
#   seq_i_end_mark: one interger, for format check, always -1
# You can use "seq_text_printer" to print the output of the generator.
@config_func
def RecurrentLayerGroupWithoutOutLinksBegin(name,
                                            in_links,
                                            seq_reversed=False,
                                            target_inlinkname=""):
    global g_current_submodel
    config_assert(g_config.model_config.type == "recurrent_nn",
                  "RecurrentLayerGroup should be used only in recurrent_nn")
    RecurrentLayerGroup(name=name)  # add to father model
    SubModelBegin(name)
    g_current_submodel.is_recurrent_layer_group = True
    g_current_submodel.reversed = seq_reversed
    g_current_submodel.target_inlinkid = -1
    in_links_count = 0
    for linkid, link in enumerate(in_links):
        if isinstance(link, basestring):
            name = link
            has_subseq = False
        else:
            name = link.link_name
            has_subseq = link.has_subseq
        # assign target_inlinkid according to target_inlinkname
        if target_inlinkname == name:
            g_current_submodel.target_inlinkid = linkid

        if in_links_count == 0:
            in_links_has_subseq = has_subseq
        else:
            config_assert(
                in_links_has_subseq == has_subseq,
                "The sequence type of in_links should be the same in RecurrentLayerGroup"
            )
        in_links_count += 1
        layer_name = MakeLayerNameInParentSubmodel(name)
        layer = g_layer_map[layer_name]
        if has_subseq:
            SequenceScatterAgentLayer(name=name, size=layer.size)
        else:
            ScatterAgentLayer(name=name, size=layer.size)

        pair = g_current_submodel.in_links.add()
        pair.layer_name = layer_name
        pair.link_name = MakeLayerNameInSubmodel(name)
        pair.has_subseq = has_subseq


@config_func
def RecurrentLayerGroupSetOutLink(link):
    if isinstance(link, basestring):
        name = link
        has_subseq = False
    else:
        name = link.link_name
        has_subseq = link.has_subseq
    layer_name = MakeLayerNameInParentSubmodel(name)
    pair = g_current_submodel.out_links.add()
    pair.layer_name = MakeLayerNameInSubmodel(name)
    pair.link_name = layer_name
    pair.has_subseq = has_subseq


def RecurrentLayerGroupSetGenerator(generator=None):
    generator.eos_layer_name = MakeLayerNameInSubmodel(generator.eos_layer_name)
    g_current_submodel.generator.CopyFrom(generator)


@config_func
def RecurrentLayerGroupBegin(name,
                             in_links,
                             out_links,
                             generator=None,
                             target_inlinkname="",
                             seq_reversed=False):
    RecurrentLayerGroupWithoutOutLinksBegin(name, in_links, seq_reversed,
                                            target_inlinkname)
    for link in out_links:
        RecurrentLayerGroupSetOutLink(link)

    if generator is not None:
        RecurrentLayerGroupSetGenerator(generator)
        config_assert(
            len(in_links) == 0, "no in_links should be passed to generator")
        config_assert(
            len(out_links) >= 1,
            "one or more than one out_links should be passed to generator")


@config_func
def RecurrentLayerGroupEnd(name):
    global g_current_submodel
    config_assert(g_current_submodel.is_recurrent_layer_group,
                  "RecurrentLayerGroup not begin")
    for pair in g_current_submodel.memories:  #check exist
        layer = g_layer_map[pair.layer_name]
        config_assert(layer is not None,
                      "memory declare wrong name:%s" % pair.layer_name)
        memory_link = g_layer_map[pair.link_name]
        config_assert(layer.size == memory_link.size,
                      "memory declare wrong size:%d" % memory_link.size)

    prev_submodel = g_current_submodel
    SubModelEnd(name)

    for pair in prev_submodel.out_links:
        layer = g_layer_map[pair.layer_name]
        # add out agent to father model
        agent_name = GetLayerBaseName(pair.link_name)
        if prev_submodel.HasField("generator"):
            DataLayer(name=agent_name, size=layer.size)
        elif pair.has_subseq:
            SequenceGatherAgentLayer(name=agent_name, size=layer.size)
        else:
            GatherAgentLayer(name=agent_name, size=layer.size)


# Define the model type
# currently, the paddle supports "nn", "recurrent_nn", "recursive_nn" and "multi_nn"
@config_func
def model_type(name):
    g_config.model_config.type = name


@config_class
class Bias(Cfg):
    def __init__(
            self,
            parameter_name=None,
            learning_rate=None,
            momentum=None,
            decay_rate=None,
            decay_rate_l1=None,
            initial_mean=None,
            initial_std=None,
            initial_strategy=None,
            initial_smart=None,
            num_batches_regularization=None,
            sparse_remote_update=None,
            gradient_clipping_threshold=None,
            is_static=None,
            is_shared=None, ):
        self.add_keys(locals())


# Define one input for a layer
@config_class
class Input(Cfg):
    def __init__(
            self,
            input_layer_name,
            parameter_name=None,
            learning_rate=None,
            momentum=None,
            decay_rate=None,
            decay_rate_l1=None,
            initial_mean=None,
            initial_std=None,
            initial_strategy=None,
            initial_smart=None,
            num_batches_regularization=None,
            sparse_remote_update=None,
            sparse_update=None,
            gradient_clipping_threshold=None,
            conv=None,
            bilinear_interp=None,
            norm=None,
            pool=None,
            image=None,
            block_expand=None,
            maxout=None,
            spp=None,
            format=None,
            nnz=None,
            is_static=None,
            is_shared=None,
            update_hooks=None,
            input_layer_argument=None, ):
        self.add_keys(locals())
        self.input_layer_name = MakeLayerNameInSubmodel(input_layer_name)


# Define a projection for iexed layer
@config_class
class Projection(Input):
    type = None  # subclass should set it correctly

    def __init__(
            self,
            input_layer_name,
            size=0,  # projection output size
            parameter_name=None,
            learning_rate=None,
            momentum=None,
            decay_rate=None,
            decay_rate_l1=None,
            initial_mean=None,
            initial_std=None,
            initial_strategy=None,
            initial_smart=None,
            num_batches_regularization=None,
            sparse_remote_update=None,
            sparse_update=None,
            gradient_clipping_threshold=None,
            ptype=None,
            format=None,
            nnz=None,
            is_static=None,
            is_shared=None,
            update_hooks=None,
            input_layer_argument=None, ):
        self.add_keys(locals())
        self.input_layer_name = MakeLayerNameInSubmodel(input_layer_name)

        self.proj_conf = ProjectionConfig()
        if ptype is not None:
            self.proj_conf.type = ptype
        else:
            self.proj_conf.type = self.type

    # calculate the output_size given input_size. return 0
    # to indicate using the size from Layer config
    def calc_output_size(self, input_layer_config):
        return self.size

    def calc_parameter_size(self, input_size, output_size):
        raise NotimplementedError

    def calc_parameter_dims(self, input_size, output_size):
        raise NotimplementedError


@config_class
class IdentityProjection(Projection):
    type = 'identity'

    def calc_output_size(self, input_layer_config):
        return input_layer_config.size

    def calc_parameter_size(self, input_size, output_size):
        return 0

    def calc_parameter_dims(self, input_size, output_size):
        return []


# Like IdentityProjection, but layer size may smaller than input size,
# the projection select dimesions [offset, offset+layer_size) from input
@config_class
class IdentityOffsetProjection(Projection):
    type = 'identity_offset'

    def __init__(self, input_layer_name, offset, **xargs):
        super(IdentityOffsetProjection, self).__init__(input_layer_name,
                                                       **xargs)
        self.proj_conf.offset = offset

    def calc_parameter_size(self, input_size, output_size):
        return 0

    def calc_parameter_dims(self, input_size, output_size):
        return []


# DotMulProjection performs element-wise multiplication with weight
@config_class
class DotMulProjection(Projection):
    type = 'dot_mul'

    def calc_output_size(self, input_layer_config):
        return input_layer_config.size

    def calc_parameter_size(self, input_size, output_size):
        return output_size

    def calc_parameter_dims(self, input_size, output_size):
        return [1, output_size]


# ScalingProjection
@config_class
class ScalingProjection(Projection):
    type = 'scaling'

    def calc_output_size(self, input_layer_config):
        return input_layer_config.size

    def calc_parameter_size(self, input_size, output_size):
        return 1

    def calc_parameter_dims(self, input_size, output_size):
        return [1, 1]


@config_class
class TableProjection(Projection):
    type = 'table'

    def calc_parameter_size(self, input_size, output_size):
        return input_size * output_size

    def calc_parameter_dims(self, input_size, output_size):
        return [input_size, output_size]


@config_class
class FullMatrixProjection(Projection):
    type = 'fc'

    def calc_parameter_size(self, input_size, output_size):
        return input_size * output_size

    def calc_parameter_dims(self, input_size, output_size):
        return [input_size, output_size]


@config_class
class TransposedFullMatrixProjection(Projection):
    type = 'trans_fc'

    def calc_parameter_size(self, input_size, output_size):
        return input_size * output_size

    def calc_parameter_dims(self, input_size, output_size):
        return [output_size, input_size]


@config_class
class ContextProjection(Projection):
    type = 'context'

    def __init__(self, input_layer_name, context_start, context_length,
                 trainable_padding, **xargs):
        super(ContextProjection, self).__init__(input_layer_name, **xargs)
        self.proj_conf.context_start = context_start
        self.proj_conf.context_length = context_length
        self.proj_conf.trainable_padding = trainable_padding
        self._total_pad = max(0, -self.proj_conf.context_start) \
                          + max(0, self.proj_conf.context_start \
                                + self.proj_conf.context_length - 1)

    def calc_output_size(self, input_layer_config):
        return input_layer_config.size * self.proj_conf.context_length

    def calc_parameter_size(self, input_size, output_size):
        if self.proj_conf.trainable_padding == False:
            return 0
        else:
            return input_size * self._total_pad

    def calc_parameter_dims(self, input_size, output_size):
        return [self._total_pad, input_size]

    _total_pad = 0


@config_class
class ConvProjection(Projection):
    type = 'conv'

    def __init__(self,
                 input_layer_name,
                 num_filters=None,
                 conv_conf=None,
                 **xargs):
        super(ConvProjection, self).__init__(input_layer_name, **xargs)

        if num_filters is not None:
            self.proj_conf.num_filters = num_filters

        parse_conv(conv_conf, input_layer_name, self.proj_conf.conv_conf,
                   num_filters)
        self.proj_conf.output_size = self.proj_conf.conv_conf.output_x * \
                                     self.proj_conf.conv_conf.output_y * \
                                     num_filters

    def calc_output_size(self, input_layer_config):
        return self.proj_conf.output_size

    def calc_parameter_size(self, input_size, output_size):
        co = self.proj_conf.num_filters
        ci = self.proj_conf.conv_conf.channels
        fh = self.proj_conf.conv_conf.filter_size
        fw = self.proj_conf.conv_conf.filter_size_y
        gr = self.proj_conf.conv_conf.groups
        return co * ci * fh * fw / gr

    def calc_bias_size(self):
        return self.proj_conf.num_filters

    def calc_parameter_dims(self, input_size, output_size):
        return None


# Define a operator for mixed layer
@config_class
class Operator(Cfg):
    type = None  # subclass should set it correctly

    def __init__(
            self,
            input_layer_names, ):
        self.add_keys(locals())
        self.operator_conf = OperatorConfig()
        self.operator_conf.type = self.type

    def check_dims(self):
        pass

    def calc_output_size(self, input_sizes):
        return 0


@config_class
class DotMulOperator(Operator):
    type = 'dot_mul'

    def __init__(self, input_layer_names, scale=None, **xargs):
        super(DotMulOperator, self).__init__(input_layer_names, **xargs)
        if scale is not None:
            self.operator_conf.dotmul_scale = scale

        config_assert(len(input_layer_names) == 2, "DotMul is binary operator")

    def check_dims(self):
        for i in range(2):
            config_assert(self.operator_conf.input_sizes[i] ==
                          self.operator_conf.output_size,
                          "DotMul input_size != output_size")

    def calc_output_size(self, input_sizes):
        return input_sizes[0]


@config_class
class ConvOperator(Operator):
    type = 'conv'

    def __init__(self,
                 input_layer_names,
                 num_filters=None,
                 conv_conf=None,
                 **xargs):
        super(ConvOperator, self).__init__(input_layer_names, **xargs)
        if num_filters is not None:
            self.operator_conf.num_filters = num_filters

        parse_conv(conv_conf,
                   MakeLayerNameInSubmodel(input_layer_names[0]),
                   self.operator_conf.conv_conf, num_filters)
        self.operator_conf.output_size = self.operator_conf.conv_conf.output_x * \
                                         self.operator_conf.conv_conf.output_y * \
                                         num_filters

        config_assert(len(input_layer_names) == 2, "Conv is binary operator")

    def calc_output_size(self, input_sizes):
        return self.operator_conf.output_size


# please refer to the comments in proto/ModelConfig.proto
@config_class
class Conv(Cfg):
    def __init__(self,
                 filter_size,
                 channels,
                 padding=None,
                 stride=None,
                 groups=None,
                 filter_channels=None,
                 output_x=None,
                 img_size=None,
                 caffe_mode=True,
                 filter_size_y=None,
                 padding_y=None,
                 stride_y=None):
        self.add_keys(locals())
        if filter_size_y is None:
            self.filter_size_y = filter_size
        if padding_y is None:
            self.padding_y = padding
        if stride_y is None:
            self.stride_y = stride
        if output_x is not None:
            config_assert(output_x <= 0)


@config_class
class BilinearInterp(Cfg):
    def __init__(self, out_size_x=None, out_size_y=None, channels=None):
        self.add_keys(locals())


@config_class
class Pool(Cfg):
    def __init__(
            self,
            pool_type,
            channels,
            size_x,
            size_y=None,
            img_width=None,
            start=None,
            stride=None,  # 1 by defalut in protobuf
            stride_y=None,
            padding=None,  # 0 by defalut in protobuf
            padding_y=None):
        self.add_keys(locals())


@config_class
class SpatialPyramidPool(Cfg):
    def __init__(self, pool_type, pyramid_height, channels):
        self.add_keys(locals())


@config_class
class Norm(Cfg):
    def __init__(self,
                 norm_type,
                 channels,
                 size,
                 scale,
                 pow,
                 output_x=None,
                 img_size=None,
                 blocked=None):
        self.add_keys(locals())


@config_class
class Image(Cfg):
    def __init__(self, channels, img_size=None):
        self.add_keys(locals())


@config_class
class BlockExpand(Cfg):
    def __init__(self,
                 channels,
                 padding_x=0,
                 padding_y=0,
                 stride_x=0,
                 stride_y=0,
                 block_x=0,
                 block_y=0,
                 img_size_x=0,
                 img_size_y=0,
                 output_x=0,
                 output_y=0):
        self.add_keys(locals())


@config_class
class MaxOut(Cfg):
    def __init__(self, channels, groups, img_size_x=0, img_size_y=0):
        self.add_keys(locals())


def DataBase(async_load_data=False,
             constant_slots=None,
             data_ratio=1,
             is_main_data=True,
             usage_ratio=None):
    # default: all sub dataproviders are treat as "main data".
    # see proto/DataConfig.proto for is_main_data
    data_config = DataConfig()

    data_config.async_load_data = async_load_data

    if constant_slots:
        data_config.constant_slots.extend(constant_slots)
    data_config.data_ratio = data_ratio
    data_config.is_main_data = is_main_data

    usage_ratio = default(usage_ratio, settings_deprecated["usage_ratio"])
    config_assert(usage_ratio >= 0 and usage_ratio <= 1,
                  "The range of usage_ratio is [0, 1]")
    data_config.usage_ratio = usage_ratio

    return data_config


@config_func
def SimpleData(files=None,
               feat_dim=None,
               context_len=None,
               buffer_capacity=None,
               **xargs):
    data_config = DataBase(**xargs)
    data_config.type = 'simple'
    data_config.files = files
    data_config.feat_dim = feat_dim
    if context_len is not None:
        data_config.context_len = context_len
    if buffer_capacity:
        data_config.buffer_capacity = buffer_capacity
    return data_config


@config_func
def PyData(files=None,
           type=None,
           file_group_queue_capacity=None,
           load_data_module=None,
           load_data_object=None,
           load_data_args="",
           load_file_count=None,
           constant_slots=None,
           load_thread_num=None,
           **xargs):
    data_config = DataBase(**xargs)
    data_config.type = 'py'
    if load_data_module in g_py_module_name_list:

        def get_path(module):
            m = __import__(load_data_module)
            return os.path.split(os.path.realpath(m.__file__))[0]

        # python C-api is not thread safe, one module can only be import once,
        # so here we nedd to copy the module with different names if it has to be
        # imported several times.
        module_new_name = "%s_copy_%d" % (load_data_module,
                                          len(g_py_module_name_list))
        g_py_module_name_list.append(module_new_name)
        module_path = "%s/%s.py" % (get_path(load_data_module),
                                    load_data_module)
        new_module_path = "%s/%s.py" % (get_path(load_data_module),
                                        module_new_name)
        if os.path.isfile(module_path) == False:
            raise Exception("File %s is not exist." % module_path)
        shutil.copy2(module_path, new_module_path)
        load_data_module = module_new_name
    else:
        g_py_module_name_list.append(load_data_module)
    if load_data_module is not None and load_data_object is not None:
        data_config.load_data_module = load_data_module
        data_config.load_data_object = load_data_object
    else:
        raise ValueError('load_data_module, load_data_object is not defined.')
    data_config.load_data_args = load_data_args

    data_config.files = files or ''
    if file_group_queue_capacity is not None:
        data_config.file_group_conf.queue_capacity = file_group_queue_capacity
    if load_file_count is not None:
        data_config.file_group_conf.load_file_count = load_file_count
    if load_thread_num is not None:
        data_config.file_group_conf.load_thread_num = load_thread_num
    if constant_slots:
        data_config.constant_slots.extend(constant_slots)
    return data_config


@config_func
def ProtoData(files=None,
              type=None,
              file_group_queue_capacity=None,
              load_file_count=None,
              constant_slots=None,
              load_thread_num=None,
              **xargs):
    data_config = DataBase(**xargs)
    if type is None:
        data_config.type = 'proto'
    else:
        data_config.type = type
    data_config.files = files

    # When type="proto_group", one data provider contains at most
    # load_file_count files, and there are at most
    # (queue_capacity + load_thread_num + 1) data providers in memory
    if file_group_queue_capacity is not None:
        data_config.file_group_conf.queue_capacity = file_group_queue_capacity
    if load_file_count is not None:
        data_config.file_group_conf.load_file_count = load_file_count
    if load_thread_num is not None:
        data_config.file_group_conf.load_thread_num = load_thread_num
    if constant_slots:
        data_config.constant_slots.extend(constant_slots)
    return data_config


#real data for training is actually provided by "sub_data" data providers.
@config_func
def MultiData(sub_data=[]):
    data_config = DataConfig()
    data_config.type = 'multi'
    data_config.sub_data_configs.extend(sub_data)
    return data_config


@config_func
def Data(type,
         files=None,
         feat_dim=None,
         slot_dims=None,
         context_len=None,
         buffer_capacity=None,
         **xargs):

    data_config = DataBase(**xargs)
    data_config.type = type
    data_config.files = files
    data_config.feat_dim = feat_dim
    data_config.slot_dims.extend(slot_dims)
    if context_len is not None:
        data_config.context_len = context_len
    data_config.buffer_capacity = buffer_capacity
    return data_config


@config_func
def TrainData(data_config, async_load_data=None):
    config_assert(not g_config.HasField('data_config'),
                  'Only one TrainData definition is allowed')
    g_config.data_config.CopyFrom(data_config)
    g_config.data_config.for_test = False
    if async_load_data is not None:
        logger.warning("Deprecated: async_load_data should be used inside"
                       " Data definition")
        g_config.data_config.async_load_data = async_load_data


@config_func
def TestData(data_config, async_load_data=None):
    config_assert(not g_config.HasField('test_data_config'),
                  'Only one TestData definition is allowed')
    g_config.test_data_config.CopyFrom(data_config)
    g_config.test_data_config.for_test = True
    if async_load_data is not None:
        logger.warning("Deprecated: async_load_data should be used inside"
                       " Data definition")
        g_config.test_data_config.async_load_data = async_load_data


#caffe_mode: compute the output size using floor instead of ceil,
#            which is consistent of caffe and CuDNN's convention.
def cnn_output_size(img_size, filter_size, padding, stride, caffe_mode):
    output = (2 * padding + img_size - filter_size) / float(stride)
    if caffe_mode:
        return 1 + int(math.floor(output))
    else:
        return 1 + int(math.ceil(output))


#calcualte image_size based on output_size for de-convolution (ConvTransLayer).
#It is the reverse function of cnn_output_size
def cnn_image_size(output_size, filter_size, padding, stride, caffe_mode):
    img_size = (output_size - 1) * stride + filter_size - 2 * padding
    if not caffe_mode:
        img_size = img_size + 1
    return img_size


def get_img_size(input_layer_name, channels):
    input = g_layer_map[input_layer_name]
    img_pixels = input.size / channels
    img_size = input.width if input.width > 0 else int(img_pixels**0.5)
    img_size_y = input.height if input.height > 0 else int(img_pixels /
                                                           img_size)
    config_assert(
        img_size * img_size_y == img_pixels,
        "Input layer %s: Incorrect input image size %d * %d for input image pixels %d"
        % (input_layer_name, img_size, img_size_y, img_pixels))
    return img_size, img_size_y


def parse_bilinear(bilinear, input_layer_name, bilinear_conf):
    parse_image(bilinear, input_layer_name, bilinear_conf.image_conf)
    bilinear_conf.out_size_x = bilinear.out_size_x
    bilinear_conf.out_size_y = bilinear.out_size_y


def parse_pool(pool, input_layer_name, pool_conf):
    pool_conf.pool_type = pool.pool_type
    config_assert(pool.pool_type in [
        'max-projection', 'avg-projection', 'cudnn-max-pool', 'cudnn-avg-pool'
    ], "pool-type %s is not in "
                  "['max-projection', 'avg-projection', "
                  "'cudnn-max-pool', 'cudnn-avg-pool']" % pool.pool_type)

    pool_conf.channels = pool.channels
    pool_conf.size_x = pool.size_x
    pool_conf.stride = pool.stride

    pool_conf.size_y = default(pool.size_y, pool_conf.size_x)
    pool_conf.stride_y = default(pool.stride_y, pool_conf.stride)

    pool_conf.img_size, pool_conf.img_size_y = \
        get_img_size(input_layer_name, pool.channels)

    config_assert(not pool.start, "start is deprecated in pooling.")

    if pool.padding is not None:
        pool_conf.padding = pool.padding
    pool_conf.padding_y = default(pool.padding_y, pool_conf.padding)
    pool_conf.output_x = cnn_output_size(pool_conf.img_size, pool_conf.size_x,
                                         pool_conf.padding, pool_conf.stride,
                                         False)
    pool_conf.output_y = cnn_output_size(pool_conf.img_size_y, pool_conf.size_y,
                                         pool_conf.padding_y,
                                         pool_conf.stride_y, False)


def parse_spp(spp, input_layer_name, spp_conf):
    parse_image(spp, input_layer_name, spp_conf.image_conf)
    spp_conf.pool_type = spp.pool_type
    config_assert(spp.pool_type in ['max-projection', 'avg-projection'],
                  "pool-type %s is not in "
                  "['max-projection', 'avg-projection']" % spp.pool_type)
    spp_conf.pyramid_height = spp.pyramid_height


def parse_image(image, input_layer_name, image_conf):
    image_conf.channels = image.channels
    image_conf.img_size, image_conf.img_size_y = \
        get_img_size(input_layer_name, image_conf.channels)


def parse_norm(norm, input_layer_name, norm_conf):
    norm_conf.norm_type = norm.norm_type
    config_assert(norm.norm_type in ['rnorm', 'cmrnorm-projection'],
                  "norm-type %s is not in [rnorm, 'cmrnorm-projection']" %
                  norm.norm_type)
    norm_conf.channels = norm.channels
    norm_conf.size = norm.size
    norm_conf.scale = norm.scale
    norm_conf.pow = norm.pow
    norm_conf.blocked = norm.blocked

    norm_conf.img_size, norm_conf.img_size_y = \
        get_img_size(input_layer_name, norm.channels)
    norm_conf.output_x = norm_conf.img_size
    norm_conf.output_y = norm_conf.img_size_y
    if norm.norm_type in ['cmrnorm-projection']:
        norm_conf.scale /= norm.size
    else:
        norm_conf.scale /= norm.size**2


#caffe_mode: compute the output size using floor instead of ceil,
#            which is consistent of caffe and CuDNN's convention.
def parse_conv(conv, input_layer_name, conv_conf, num_filters, trans=False):
    conv_conf.filter_size = conv.filter_size
    conv_conf.filter_size_y = conv.filter_size_y
    conv_conf.channels = conv.channels
    conv_conf.padding = conv.padding
    conv_conf.padding_y = conv.padding_y
    conv_conf.stride = conv.stride
    conv_conf.stride_y = conv.stride_y
    conv_conf.groups = conv.groups
    conv_conf.caffe_mode = conv.caffe_mode

    if not trans:
        conv_conf.filter_channels = conv.channels / conv.groups
        conv_conf.img_size, conv_conf.img_size_y = \
            get_img_size(input_layer_name, conv.channels)
        conv_conf.output_x = cnn_output_size(
            conv_conf.img_size, conv_conf.filter_size, conv_conf.padding,
            conv_conf.stride, conv_conf.caffe_mode)
        conv_conf.output_y = cnn_output_size(
            conv_conf.img_size_y, conv_conf.filter_size_y, conv_conf.padding_y,
            conv_conf.stride_y, conv_conf.caffe_mode)
    else:
        conv_conf.filter_channels = num_filters / conv.groups
        conv_conf.output_x, conv_conf.output_y = \
            get_img_size(input_layer_name, conv.channels)
        conv_conf.img_size = cnn_image_size(
            conv_conf.output_x, conv_conf.filter_size, conv_conf.padding,
            conv_conf.stride, conv_conf.caffe_mode)
        conv_conf.img_size_y = cnn_image_size(
            conv_conf.output_y, conv_conf.filter_size_y, conv_conf.padding_y,
            conv_conf.stride_y, conv_conf.caffe_mode)


def parse_block_expand(block_expand, input_layer_name, block_expand_conf):
    block_expand_conf.channels = block_expand.channels
    block_expand_conf.stride_x = block_expand.stride_x
    block_expand_conf.stride_y = block_expand.stride_y
    block_expand_conf.padding_x = block_expand.padding_x
    block_expand_conf.padding_y = block_expand.padding_y
    block_expand_conf.block_x = block_expand.block_x
    block_expand_conf.block_y = block_expand.block_y
    block_expand_conf.img_size_x = block_expand.img_size_x
    block_expand_conf.img_size_y = block_expand.img_size_y
    if block_expand_conf.img_size_x == 0:
        block_expand_conf.output_x = 0
    else:
        block_expand_conf.output_x = cnn_output_size(
            block_expand.img_size_x, block_expand.block_x,
            block_expand.padding_x, block_expand.stride_x, False)

    if block_expand_conf.img_size_y == 0:
        block_expand_conf.output_y = 0
    else:
        block_expand_conf.output_y = cnn_output_size(
            block_expand.img_size_y, block_expand.block_y,
            block_expand.padding_y, block_expand.stride_y, False)


def parse_maxout(maxout, input_layer_name, maxout_conf):
    parse_image(maxout, input_layer_name, maxout_conf.image_conf)
    maxout_conf.groups = maxout.groups


# Define an evaluator
@config_func
def Evaluator(
        name,
        type,
        inputs,
        chunk_scheme=None,
        num_chunk_types=None,
        classification_threshold=None,
        positive_label=None,
        dict_file=None,
        result_file=None,
        num_results=None,
        delimited=None, ):
    evaluator = g_config.model_config.evaluators.add()
    evaluator.type = type
    evaluator.name = MakeLayerNameInSubmodel(name)
    if type_of(inputs) == str:
        inputs = [inputs]

    evaluator.input_layers.extend(
        [MakeLayerNameInSubmodel(name) for name in inputs])

    if chunk_scheme is not None:
        evaluator.chunk_scheme = chunk_scheme
        evaluator.num_chunk_types = num_chunk_types
    g_current_submodel.evaluator_names.append(evaluator.name)

    if classification_threshold is not None:
        evaluator.classification_threshold = classification_threshold
    if positive_label is not None:
        evaluator.positive_label = positive_label
    if dict_file is not None:
        evaluator.dict_file = dict_file

    if result_file is not None:
        evaluator.result_file = result_file
    if num_results is not None:
        evaluator.num_results = num_results
    if delimited is not None:
        evaluator.delimited = delimited


class LayerBase(object):
    def __init__(
            self,
            name,
            type,
            size,  # size can be 0. In this case, subclass should set it.
            inputs,
            device=None,
            active_type="",
            drop_rate=0.,
            coeff=None):
        config_assert('@' not in name,
                      "layer name: %s contain special character @" % name)
        global g_current_submodel
        name = MakeLayerNameInSubmodel(name)

        config_assert(name not in g_layer_map,
                      'Duplicated layer name: %s' % name)

        self.inputs = copy.deepcopy(inputs)
        self.operators = []

        if self.inputs is None:
            self.inputs = []
        elif type_of(self.inputs) != list:
            self.inputs = [self.inputs]

        self.config = g_config.model_config.layers.add()
        assert isinstance(self.config, LayerConfig)
        self.config.name = name
        self.config.type = type
        self.config.active_type = active_type
        if coeff is not None:
            self.config.coeff = float(coeff)
        if size != 0:
            self.config.size = size
        if drop_rate != 0:
            self.config.drop_rate = drop_rate

        if device is not None:
            self.config.device = device
        elif g_default_device is not None:
            self.config.device = g_default_device

        for input_index in xrange(len(self.inputs)):
            input = self.inputs[input_index]
            input_config = None
            input_layer_name = ''
            if type_of(input) == str:
                input_layer_name = input
                input_config = Input(
                    input_layer_name=input,
                    parameter_name=gen_parameter_name(name, input_index))
                input_layer_name = input_config.input_layer_name
            elif isinstance(input, Input):
                input_layer_name = input.input_layer_name
                input_config = input
                if input_config.parameter_name is None:
                    input_config.parameter_name = \
                        gen_parameter_name(name, input_index)
            elif isinstance(input, Operator):
                self.operators.append(input)
                input.operator_conf.input_indices.append(input_index)
                input_config = Input(input.input_layer_names[0])
                input_layer_name = input_config.input_layer_name
            else:
                raise ValueError('Wrong type for inputs: %s' % type_of(input))
            config_assert(input_layer_name in g_layer_map,
                          "Unknown input layer '%s' for layer %s" %
                          (input_layer_name, name))
            self.inputs[input_index] = input_config
            layer_input = self.config.inputs.add()
            layer_input.input_layer_name = input_config.input_layer_name
            if input_config.input_layer_argument is not None:
                layer_input.input_layer_argument = \
                    input_config.input_layer_argument

        g_layer_map[name] = self.config

        g_current_submodel.layer_names.append(self.config.name)

        if self.config.type != 'data' and g_pass_height_width:
            height = self.get_input_layer(0).height
            width = self.get_input_layer(0).width
            if height and width:
                self.set_layer_height_width(height, width)

    def get_input_layer(self, input_index):
        return g_layer_map[self.config.inputs[input_index].input_layer_name]

    # will return the bias created if not *for_self*
    def create_bias_parameter(
            self,
            bias,  # True/False or BiasCfg
            size,
            dims=None,
            for_self=True,  # whether create bias for layer self
    ):

        if size == 0:
            return
        if dims is None:
            dims = [1, size]

        config_assert(
            type_of(bias) == bool or type_of(bias) == Bias,
            'Incorrect type for bias: %s' % type_of(bias))

        if type_of(bias) == bool:
            if bias:
                bias = Bias()

        if type_of(bias) == Bias:
            if bias.parameter_name is None:
                bias.parameter_name = gen_bias_parameter_name(self.config.name)
            if bias.parameter_name not in g_parameter_map:
                assert isinstance(self.config, LayerConfig)

                Parameter(
                    bias.parameter_name,
                    size,
                    self.config.device
                    if self.config.HasField('device') else None,
                    dims,
                    bias.learning_rate,
                    bias.momentum,
                    decay_rate=bias.decay_rate,
                    decay_rate_l1=bias.decay_rate_l1,
                    initial_mean=bias.initial_mean,
                    initial_std=bias.initial_std,
                    initial_strategy=bias.initial_strategy,
                    initial_smart=bias.initial_smart,
                    num_batches_regularization=bias.num_batches_regularization,
                    sparse_remote_update=bias.sparse_remote_update,
                    gradient_clipping_threshold=bias.
                    gradient_clipping_threshold,
                    is_static=bias.is_static,
                    is_shared=bias.is_shared, )
            if for_self:
                self.config.bias_parameter_name = bias.parameter_name
            else:
                return bias.parameter_name

    def create_input_parameter(self,
                               input_index,
                               size,
                               dims=None,
                               sparse=None,
                               format=None):
        if dims is None:
            # TODO(yuyang18): print warning and callstack here!
            dims = list()

        if size == 0:
            return

        input_config = self.inputs[input_index]

        self.config.inputs[input_index].input_parameter_name = \
            input_config.parameter_name

        if input_config.parameter_name in g_parameter_map:
            para = g_parameter_map[input_config.parameter_name]
            config_assert(size == para.size, (
                'Shared parameter "%s" does not ' + 'have same size: %s vs. %s')
                          % (input_config.parameter_name, para.size, size))

            config_assert(dims == para.dims, (
                'Shared parameter "%s" does not ' + 'have same dims: %s vs. %s')
                          % (input_config.parameter_name, para.dims, dims))
            return

        Parameter(
            input_config.parameter_name,
            size,
            self.config.device if self.config.HasField("device") else None,
            dims,
            input_config.learning_rate,
            input_config.momentum,
            decay_rate=input_config.decay_rate,
            decay_rate_l1=input_config.decay_rate_l1,
            initial_mean=input_config.initial_mean,
            initial_std=input_config.initial_std,
            initial_strategy=input_config.initial_strategy,
            initial_smart=input_config.initial_smart,
            num_batches_regularization=input_config.num_batches_regularization,
            sparse_remote_update=input_config.sparse_remote_update,
            sparse_update=input_config.sparse_update,
            gradient_clipping_threshold=input_config.
            gradient_clipping_threshold,
            sparse=sparse,
            format=format,
            is_static=input_config.is_static,
            is_shared=input_config.is_shared,
            update_hooks=input_config.update_hooks)

    def set_layer_size(self, size):
        if self.config.size == 0:
            self.config.size = size
        else:
            config_assert(self.config.size == size,
                          'Different inputs result in' +
                          'different layer size at layer %s' % self.config.name)

    def set_layer_height_width(self, height, width):
        self.config.height = height
        self.config.width = width

    def set_cnn_layer(self,
                      input_layer_name,
                      height,
                      width,
                      channels,
                      is_print=True):
        size = height * width * channels
        self.set_layer_size(size)
        self.set_layer_height_width(height, width)
        if is_print:
            print("output for %s: c = %d, h = %d, w = %d, size = %d" %
                  (input_layer_name, channels, height, width, size))


@config_layer('multi_class_cross_entropy_with_selfnorm')
class MultiClassCrossEntropySelfNormCostLayer(LayerBase):
    def __init__(self, name, inputs, softmax_selfnorm_alpha=0.1, **xargs):
        super(MultiClassCrossEntropySelfNormCostLayer, self).__init__(
            name, 'multi_class_cross_entropy_with_selfnorm', 0, inputs, **xargs)
        self.config.softmax_selfnorm_alpha = softmax_selfnorm_alpha


@config_layer('fc')
class FCLayer(LayerBase):
    def __init__(self, name, size, inputs, bias=True, **xargs):
        super(FCLayer, self).__init__(name, 'fc', size, inputs=inputs, **xargs)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            psize = self.config.size * input_layer.size
            dims = [input_layer.size, self.config.size]
            format = self.inputs[input_index].format
            sparse = format == "csr" or format == "csc"

            if sparse:
                psize = self.inputs[input_index].nnz
            else:
                sparse = None

            self.create_input_parameter(input_index, psize, dims, sparse,
                                        format)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('selective_fc')
class SelectiveFCLayer(LayerBase):
    def __init__(self,
                 name,
                 size,
                 inputs,
                 bias=True,
                 selective_fc_pass_generation=False,
                 has_selected_colums=True,
                 selective_fc_full_mul_ratio=0.02,
                 selective_fc_parallel_plain_mul_thread_num=None,
                 **xargs):
        super(SelectiveFCLayer, self).__init__(
            name, 'selective_fc', size, inputs=inputs, **xargs)
        # user MUST know if selctive fc is used in training,
        # parameter matrices saved by this layer are automatically transposed,
        # BUT bias is not.

        # if selective_fc is used only in testing mode, and parameters for
        # this layer are trained by fully connected layers,
        # then TranposedFullMatrixProjectin MUST be used in training
        # to avoid manual transpose in testing.

        self.config.selective_fc_pass_generation = selective_fc_pass_generation
        self.config.has_selected_colums = has_selected_colums
        self.config.selective_fc_full_mul_ratio = selective_fc_full_mul_ratio
        if selective_fc_parallel_plain_mul_thread_num is not None:
            self.config.selective_fc_parallel_plain_mul_thread_num = selective_fc_parallel_plain_mul_thread_num

        input_num = len(self.inputs)
        if has_selected_colums:
            config_assert(input_num >= 2,
                          ("if indices of selected columns are not specified, "
                           "selective_fc Layer has at least two inputs"))
            input_num -= 1

        for input_index in xrange(input_num):
            input_layer = self.get_input_layer(input_index)
            psize = self.config.size * input_layer.size
            dims = [input_layer.size, self.config.size]
            dims = dims[::-1]  # transpose the parameter
            format = self.inputs[input_index].format
            sparse = format == "csr" or format == "csc"
            if sparse:
                psize = self.inputs[input_index].nnz

            self.create_input_parameter(input_index, psize, dims, sparse,
                                        format)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('print')
class PrintLayer(LayerBase):
    def __init__(self, name, inputs):
        super(PrintLayer, self).__init__(name, 'print', 0, inputs)


@config_layer('data')
class DataLayer(LayerBase):
    def __init__(self, name, size, height=None, width=None, device=None):
        super(DataLayer, self).__init__(
            name, 'data', size, inputs=[], device=device)
        if height and width:
            self.set_layer_height_width(height, width)


'''
DataNormLayer: A layer for data normalization
Input: One and only one input layer is accepted. The input layer must
       be DataLayer with dense data type
Output: The normalization of the input data

Reference:
    LA Shalabi, Z Shaaban, B Kasasbeh. Data mining: A preprocessing engine

Example:
    Layer(
        name = "norm_input_layer",
        type = "data_norm",
        inputs = [Input("input_layer",
                        parameter_name = "_slot0.stats")],
        data_norm_strategy = "z-score",
    )

Note:
  (1) The parameter has been calculated in the preprocessing stage,
      and should be initialized by --init_model_path when training.
  (2) Three data normalization methoeds are considered
          z-score: y = (x-mean)/std
          min-max: y = (x-min)/(max-min)
          decimal-scaling: y = x/10^j, where j is the smallest integer such that max(|y|)<1
'''


@config_layer('data_norm')
class DataNormLayer(LayerBase):
    def __init__(self, name, inputs, data_norm_strategy="z-score", device=None):
        super(DataNormLayer, self).__init__(
            name, 'data_norm', 0, inputs=inputs, device=device)
        self.config.data_norm_strategy = data_norm_strategy
        config_assert(len(inputs) == 1, 'DataNormLayer must have 1 input')
        input_layer = self.get_input_layer(0)
        self.set_layer_size(input_layer.size)
        para_size = 5 * input_layer.size
        para_dims = [5, input_layer.size]
        self.inputs[0].is_static = True
        self.create_input_parameter(0, para_size, para_dims)


@config_layer('prelu')
class ParameterReluLayer(LayerBase):
    layer_type = 'prelu'

    def __init__(self, name, inputs, partial_sum=1, **args):
        super(ParameterReluLayer, self).__init__(
            name, self.layer_type, 0, inputs=inputs, **args)
        config_assert(len(self.inputs) == 1)
        config_assert(self.input_layer.size % partial_sum == 0)
        input_layer = self.get_input_layer(0)
        self.set_layer_size(input_layer.size)
        self.create_input_parameter(0, input_layer.size / partial_sum)


@config_layer('conv')
class ConvLayerBase(LayerBase):
    layer_type = 'conv'

    def __init__(self,
                 name,
                 inputs=[],
                 bias=True,
                 num_filters=None,
                 shared_biases=False,
                 **xargs):
        super(ConvLayerBase, self).__init__(
            name, self.layer_type, 0, inputs=inputs, **xargs)

        if num_filters is not None:
            self.config.num_filters = num_filters

        use_gpu = int(g_command_config_args.get("use_gpu", 0))
        parallel_nn = int(g_command_config_args.get("parallel_nn", 0))

        # Automatically select cudnn_type for GPU and exconv for CPU
        # if set type=conv, but still reserve the way user specify
        # exconv or cudnn_conv manually.
        if self.layer_type == "cudnn_conv":
            config_assert(use_gpu, "cudnn_conv only support GPU")

        if (use_gpu == 1 and self.layer_type != "exconv" and
            (parallel_nn == 0 or self.config.device > -1)):
            self.layer_type = "cudnn_conv"
        else:
            self.layer_type = "exconv"
        # need to specify layer in config
        self.config.type = self.layer_type

        if shared_biases is not None:
            self.config.shared_biases = shared_biases

        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            conv_conf = self.config.inputs[input_index].conv_conf
            parse_conv(self.inputs[input_index].conv, input_layer.name,
                       conv_conf, num_filters)
            psize = self.calc_parameter_size(conv_conf)
            self.create_input_parameter(input_index, psize)
            self.set_cnn_layer(name, conv_conf.output_y, conv_conf.output_x,
                               self.config.num_filters)

        psize = self.config.size
        if shared_biases:
            psize = self.config.num_filters
        self.create_bias_parameter(bias, psize, [psize, 1])

    def calc_parameter_size(self, conv_conf):
        return self.config.num_filters * conv_conf.filter_channels \
                    * (conv_conf.filter_size * conv_conf.filter_size_y)


@config_layer('exconv')
class ConvLayer(ConvLayerBase):
    layer_type = 'exconv'


@config_layer('cudnn_conv')
class ConvLayer(ConvLayerBase):
    layer_type = 'cudnn_conv'


@config_layer('convt')
class ConvTransLayerBase(LayerBase):
    layer_type = 'convt'

    def __init__(self,
                 name,
                 inputs=[],
                 bias=True,
                 num_filters=None,
                 shared_biases=False,
                 **xargs):
        super(ConvTransLayerBase, self).__init__(
            name, self.layer_type, 0, inputs=inputs, **xargs)

        if num_filters is not None:
            self.config.num_filters = num_filters

        use_gpu = int(g_command_config_args.get("use_gpu", 0))
        parallel_nn = int(g_command_config_args.get("parallel_nn", 0))

        # cudnn_convt has not been implemented so use exconvt only
        self.layer_type = "exconvt"
        # need to specify layer in config
        self.config.type = self.layer_type

        if shared_biases is not None:
            self.config.shared_biases = shared_biases

        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            parse_conv(
                self.inputs[input_index].conv,
                input_layer.name,
                self.config.inputs[input_index].conv_conf,
                num_filters,
                trans=True)
            conv_conf = self.config.inputs[input_index].conv_conf
            psize = self.calc_parameter_size(conv_conf)
            print("output size for %s is %d " % (name, conv_conf.output_x))
            self.create_input_parameter(input_index, psize)
            self.set_layer_size(
                (conv_conf.img_size**2) * self.config.num_filters)

        psize = self.config.size
        if shared_biases:
            psize = self.config.num_filters
        self.create_bias_parameter(bias, psize, [psize, 1])

    def calc_parameter_size(self, conv_conf):
        return conv_conf.channels * conv_conf.filter_channels \
                    * (conv_conf.filter_size * conv_conf.filter_size_y)


@config_layer('exconvt')
class ConvTransLayer(ConvTransLayerBase):
    layer_type = 'exconvt'


@config_layer('norm')
class NormLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(NormLayer, self).__init__(
            name, 'norm', 0, inputs=inputs, device=device)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            norm_conf = self.config.inputs[input_index].norm_conf
            parse_norm(self.inputs[input_index].norm, input_layer.name,
                       norm_conf)
            self.set_cnn_layer(name, norm_conf.output_y, norm_conf.output_x,
                               norm_conf.channels, False)


@config_layer('pool')
class PoolLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(PoolLayer, self).__init__(
            name, 'pool', 0, inputs=inputs, device=device)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            pool_conf = self.config.inputs[input_index].pool_conf
            parse_pool(self.inputs[input_index].pool, input_layer.name,
                       pool_conf)
            self.set_cnn_layer(name, pool_conf.output_y, pool_conf.output_x,
                               pool_conf.channels)


@config_layer('spp')
class SpatialPyramidPoolLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(SpatialPyramidPoolLayer, self).__init__(
            name, 'spp', 0, inputs=inputs, device=device)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            spp_conf = self.config.inputs[input_index].spp_conf
            parse_spp(self.inputs[input_index].spp, input_layer.name, spp_conf)
            output_x = (pow(4, spp_conf.pyramid_height) - 1) / (4 - 1)
            self.set_cnn_layer(name, 1, output_x, spp_conf.image_conf.channels)


@config_layer('batch_norm')
class BatchNormLayer(LayerBase):
    layer_type = 'batch_norm'

    def __init__(self,
                 name,
                 inputs,
                 active_type="linear",
                 bias=True,
                 device=None,
                 use_global_stats=True,
                 moving_average_fraction=0.9,
                 batch_norm_type=None,
                 **xargs):
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]
        config_assert(
            len(inputs) == 1, "BatchNormLayer must have one and only one input")
        # Create Input for moving mean and std,
        # in batch normalization layer.
        # These paras no need to update, so set is_static is true.
        # If not use is_static, even set learning_rate = 0, decay_rate = 0,
        # these paras will change if set average_window in configure.
        use_gpu = bool(int(g_command_config_args.get("use_gpu", 0)))
        is_shared = True if not use_gpu else False
        for i in xrange(2):
            inputs.append(
                Input(
                    inputs[0].input_layer_name,
                    initial_std=0.0,
                    initial_mean=0.0,
                    is_static=True,
                    is_shared=is_shared, ))

        parallel_nn = bool(int(g_command_config_args.get("parallel_nn", 0)))
        cudnn_version = int(g_command_config_args.get("cudnn_version", 0))
        # Automatically select cudnn_batch_norm for GPU and batch_norm for CPU.
        # Also based on cudnn version.
        use_cudnn = use_gpu and batch_norm_type != "batch_norm" and \
            ((not parallel_nn) or self.config.device > -1) and \
            cudnn_version >= 4007
        self.layer_type = "cudnn_batch_norm" if use_cudnn else "batch_norm"
        super(BatchNormLayer, self).__init__(
            name,
            self.layer_type,
            0,
            active_type=active_type,
            inputs=inputs,
            device=device,
            **xargs)

        if use_global_stats is not None:
            self.config.use_global_stats = use_global_stats
        if moving_average_fraction is not None:
            self.config.moving_average_fraction = moving_average_fraction

        input_layer = self.get_input_layer(0)
        image_conf = self.config.inputs[0].image_conf
        parse_image(self.inputs[0].image, input_layer.name, image_conf)
        self.set_cnn_layer(name, image_conf.img_size_y, image_conf.img_size,
                           image_conf.channels, False)

        psize = self.calc_parameter_size(image_conf)
        dims = [1, psize]
        self.create_input_parameter(0, psize)
        self.create_input_parameter(1, psize, dims)
        self.create_input_parameter(2, psize, dims)

        self.create_bias_parameter(bias, psize)

    def calc_parameter_size(self, image_conf):
        return image_conf.channels


@config_layer('trans')
class TransLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(TransLayer, self).__init__(
            name, 'trans', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 1,
            'TransLayer must have one and only one input')
        self.set_layer_size(self.get_input_layer(0).size)


@config_layer('resize')
class ResizeLayer(LayerBase):
    def __init__(self, name, size, inputs, device=None):
        super(ResizeLayer, self).__init__(
            name, 'resize', size=size, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 1,
            'ResizeLayer must have one and only one input')


@config_layer('blockexpand')
class BlockExpandLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(BlockExpandLayer, self).__init__(
            name, 'blockexpand', 0, inputs=inputs, device=device)
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            parse_block_expand(
                self.inputs[input_index].block_expand, input_layer.name,
                self.config.inputs[input_index].block_expand_conf)
            block_expand_conf = self.config.inputs[
                input_index].block_expand_conf
            self.set_layer_size(block_expand_conf.block_x *
                                block_expand_conf.block_y *
                                block_expand_conf.channels)


@config_layer('maxout')
class MaxOutLayer(LayerBase):
    def __init__(self, name, inputs, **xargs):
        super(MaxOutLayer, self).__init__(
            name, 'maxout', 0, inputs=inputs, **xargs)
        input_layer = self.get_input_layer(0)
        maxout_conf = self.config.inputs[0].maxout_conf
        parse_maxout(self.inputs[0].maxout, input_layer.name, maxout_conf)
        out_channels = maxout_conf.image_conf.channels / maxout_conf.groups
        self.set_cnn_layer(name, g_layer_map[input_layer.name].height,
                           g_layer_map[input_layer.name].width, out_channels)


# key: cost type
# value: cost class
g_cost_map = {}


# define a cost layer without any parameters
def define_cost(class_name, cost_type):
    def init(cls, name, inputs, device=None, coeff=1.):
        super(type(cls), cls).__init__(
            name, cost_type, 1, inputs, device=device, coeff=coeff)

    cls = type(class_name, (LayerBase, ), dict(__init__=init))
    global g_cost_map
    g_cost_map[cost_type] = cls


define_cost('MultiClassCrossEntropy', 'multi-class-cross-entropy')
define_cost('RankingCost', 'rank-cost')
define_cost('AucValidation', 'auc-validation')
define_cost('PnpairValidation', 'pnpair-validation')
define_cost('SumOfSquaresCostLayer', 'square_error')
define_cost('MultiBinaryLabelCrossEntropy', 'multi_binary_label_cross_entropy')
define_cost('SoftBinaryClassCrossEntropy', 'soft_binary_class_cross_entropy')
define_cost('HuberTwoClass', 'huber')
define_cost('SumCost', 'sum_cost')


@config_layer('hsigmoid')
class HierarchicalSigmoidLayer(LayerBase):
    def __init__(self, name, num_classes, inputs, device=None, bias=True):
        super(HierarchicalSigmoidLayer, self).__init__(
            name, 'hsigmoid', 1, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) >= 2,
            'HierarchicalSigmoidLayer must have at least 2 inputs')
        self.config.num_classes = num_classes
        for input_index in xrange(len(self.inputs) - 1):
            input_layer = self.get_input_layer(input_index)
            psize = (num_classes - 1) * input_layer.size
            dims = [num_classes - 1, input_layer.size]
            self.create_input_parameter(input_index, psize, dims)
        self.create_bias_parameter(bias, num_classes - 1)


'''
lambdaCost for lambdaRank LTR approach

Usage:
  Example: Layer(name = "cost", type = "lambda_cost", NDCG_num = 8,
             max_sort_size = -1, inputs = ["output", "score"])

  Input data: Samples of the same query should be loaded as a sequence,
          by ProtoDataProvider or PyDataProvider etc.. User should provide
          scores for each sample. The score slot should be the 2nd
          input of lambdaRank layer.

  NDCG_num = the size of NDCG, e.g., 5 for NDCG@5.
    Note: NDCG_num must be less than or equal to the minimum
          size of lists.

  max_sort_size = the size of partial sorting in calculating gradient.
    Note: If max_sort_size = -1, then for each list, the algorithm will
          sort the entire list to get gradient.
          In other cases, max_sort_size must be greater than or equal
          to NDCG_num.
          max_sort_size can be greater than the size of a list, in which
          case the algorithm will sort the entire list to get gradient.
'''


@config_layer('lambda_cost')
class LambdaCost(LayerBase):
    def __init__(self, name, inputs, NDCG_num=5, max_sort_size=-1, device=None):
        super(LambdaCost, self).__init__(
            name, 'lambda_cost', 1, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 2, 'lambdaCost must have 2 inputs')
        self.config.NDCG_num = NDCG_num
        if max_sort_size != -1:
            config_assert(
                NDCG_num <= max_sort_size,
                'NDCG_num must be less than or equal to max_sort_size')
        self.config.max_sort_size = max_sort_size


@config_layer('nce')
class NCELayer(LayerBase):
    def __init__(self,
                 name,
                 num_classes,
                 inputs,
                 num_neg_samples=10,
                 neg_sampling_dist=None,
                 bias=True,
                 **xargs):
        super(NCELayer, self).__init__(name, 'nce', 1, inputs=inputs, **xargs)
        config_assert(
            len(self.inputs) >= 2, 'NCELayer must have at least 2 inputs')
        self.config.num_classes = num_classes
        if neg_sampling_dist is not None:
            config_assert(
                len(neg_sampling_dist) == num_classes,
                'len(neg_sampling_dist)(%s) is not same as num_classes (%s)' %
                (len(neg_sampling_dist), num_classes))
            s = sum(neg_sampling_dist)
            config_assert(
                abs(s - 1) < 1e-5,
                'The sum of neg_sampling_dist (%s) is not 1' % s)

            self.config.neg_sampling_dist.extend(neg_sampling_dist)

        self.config.num_neg_samples = num_neg_samples
        num_real_inputs = len(self.inputs) - 1
        input_layer = self.get_input_layer(num_real_inputs)
        config_assert(input_layer.type == 'data',
                      'Expecting the last input layer of an nce layer to be '
                      'a data layer')

        if (num_real_inputs > 1 and input_layer.size == 1 and
                self.get_input_layer(num_real_inputs - 1).type == 'data'):
            # This input layer is assumed to be a sample weight layer
            num_real_inputs -= 1

        for input_index in xrange(num_real_inputs):
            input_layer = self.get_input_layer(input_index)
            psize = num_classes * input_layer.size
            dims = [num_classes, input_layer.size]
            self.create_input_parameter(input_index, psize, dims)
        self.create_bias_parameter(bias, num_classes)


@config_layer('addto')
class AddToLayer(LayerBase):
    def __init__(self, name, inputs, bias=True, **xargs):
        super(AddToLayer, self).__init__(
            name, 'addto', 0, inputs=inputs, **xargs)
        config_assert(len(inputs) > 0, 'inputs cannot be empty for AddToLayer')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('agent')
class AgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(AgentLayer, self).__init__(
            name, 'agent', size, inputs=[], device=device)


@config_layer('sequence_agent')
class SequenceAgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(SequenceAgentLayer, self).__init__(
            name, 'sequence_agent', size, inputs=[], device=device)


@config_layer('gather_agent')
class GatherAgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(GatherAgentLayer, self).__init__(
            name, 'gather_agent', size, inputs=[], device=device)


@config_layer('scatter_agent')
class ScatterAgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(ScatterAgentLayer, self).__init__(
            name, 'scatter_agent', size, inputs=[], device=device)


@config_layer('sequence_gather_agent')
class SequenceGatherAgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(SequenceGatherAgentLayer, self).__init__(
            name, 'sequence_gather_agent', size, inputs=[], device=device)


@config_layer('sequence_scatter_agent')
class SequenceScatterAgentLayer(LayerBase):
    def __init__(self, name, size, device=None):
        super(SequenceScatterAgentLayer, self).__init__(
            name, 'sequence_scatter_agent', size, inputs=[], device=device)


@config_layer('multiplex')
class MultiplexLayer(LayerBase):
    def __init__(self, name, inputs, size, device=None):
        super(MultiplexLayer, self).__init__(
            name, 'multiplex', size, inputs=inputs, device=device)
        config_assert(
            len(inputs) > 2, 'MultiplexLayer should have more than 2 inputs.')
        for i in range(1, len(inputs)):
            config_assert(
                self.get_input_layer(i).size == size,
                "All the input layers except the first one should"
                "have the same size as the MultiplexLayer.")


@config_func
def Link(
        name,
        has_subseq=False, ):
    link_config = LinkConfig()
    link_config.link_name = name
    link_config.has_subseq = has_subseq
    return link_config


# memory for recurrent layer group.
# *name* and *size* are actual layer's name and size.
# will return name of the memory,
# use this name if you assign the memory as other layer's input
#
# boot frame of memory is zeroed by default,
# or initialize by boot layer output if *boot_layer* set,
# or initialize by trainable bias if *boot_bias* set,
# or initialize by a constant id if *boot_with_const_id* set
#
# Memory can be a sequence if *is_sequence* set, this type of memory
# can only be initailized by a *boot_layer* which is a sequence.
#
@config_func
def Memory(
        name,
        size,
        is_sequence=False,
        boot_layer=None,
        boot_bias=False,
        boot_bias_active_type="",
        boot_with_const_id=None, ):
    agent_name = name + "+delay1"
    if is_sequence:
        agent_layer = SequenceAgentLayer(agent_name, size)
    else:
        agent_layer = AgentLayer(agent_name, size)
    config_assert(g_current_submodel.is_recurrent_layer_group,
                  'Memory should be used in recurrent layer group only')
    memory = g_current_submodel.memories.add()
    memory.layer_name = MakeLayerNameInSubmodel(name)
    memory.link_name = MakeLayerNameInSubmodel(agent_name)
    memory.is_sequence = is_sequence
    options = sum((boot_layer is not None, bool(boot_bias),
                   boot_with_const_id is not None))
    config_assert(
        options <= 1,
        'take one option at most from boot_layer, boot_bias, or boot_with_const_id'
    )
    if boot_layer is not None:
        boot_layer = MakeLayerNameInParentSubmodel(boot_layer)
        config_assert(boot_layer in g_layer_map,
                      'boot_layer "%s" does not correspond to a layer name' %
                      boot_layer)
        memory.boot_layer_name = boot_layer
    elif boot_bias:
        memory.boot_bias_parameter_name = agent_layer.create_bias_parameter(
            boot_bias, size, for_self=False)
        memory.boot_bias_active_type = boot_bias_active_type
    elif boot_with_const_id is not None:
        memory.boot_with_const_id = boot_with_const_id
    return agent_name


# Generator for recurrent layer group, to use it:
#  1. define a id layer as output of layer group
#  2. define a memory of this id layer, and assign a boot id(begin of sequence)
#  3. define a eos check layer and fill its name in generator's *eos_layer_name*
# Sequence generation will stop when eos check return 1 or *max_num_frames* reached.
# If *beam_size* is greater than one, generator will use beam search.
#   in beam search, if *num_results_per_sample* set, one sample sequence can output
#   multiple results each with a probility.
@config_func
def Generator(
        max_num_frames,
        eos_layer_name="eos_check",
        num_results_per_sample=1,
        beam_size=1,
        log_prob=None, ):
    generator_config = GeneratorConfig()
    generator_config.max_num_frames = max_num_frames
    generator_config.eos_layer_name = eos_layer_name
    generator_config.num_results_per_sample = num_results_per_sample
    generator_config.beam_size = beam_size
    if log_prob is not None:
        generator_config.log_prob = log_prob
    return generator_config


@config_layer('expand')
class ExpandLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 trans_type='non-seq',
                 device=None,
                 bias=False):
        super(ExpandLayer, self).__init__(
            name, 'expand', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 2, 'ExpandLayer takes 2 and only 2 inputs')
        self.config.trans_type = trans_type
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
        self.set_layer_size(self.get_input_layer(0).size)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('featmap_expand')
class FeatMapExpandLayer(LayerBase):
    def __init__(self, name, inputs, device=None, num_filters=None, bias=False):
        super(FeatMapExpandLayer, self).__init__(
            name, 'featmap_expand', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 1, 'ExpandLayer takes 1 and only 1 inputs')
        if num_filters is not None:
            self.config.num_filters = num_filters
        else:
            logger.fatal("FeatMapExpandLayer must specify num_filters.")
        self.set_layer_size(self.get_input_layer(0).size * num_filters)


@config_layer('max')
class MaxLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 trans_type='non-seq',
                 active_type='linear',
                 device=None,
                 bias=False,
                 output_max_index=None):
        super(MaxLayer, self).__init__(
            name, 'max', 0, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 1, 'MaxLayer must have 1 input')
        self.config.trans_type = trans_type
        self.config.active_type = active_type
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)
        self.create_bias_parameter(bias, self.config.size)
        if output_max_index is not None:
            self.config.output_max_index = output_max_index


@config_layer('maxid')
class MaxIdLayer(LayerBase):
    def __init__(self, name, inputs, beam_size=None, device=None):
        super(MaxIdLayer, self).__init__(
            name, 'maxid', 0, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 1, 'MaxIdLayer must have 1 input')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)

        if beam_size is None:
            global g_current_submodel
            if g_current_submodel.HasField("generator"):
                self.config.beam_size = g_current_submodel.generator.beam_size
        else:
            self.config.beam_size = beam_size


@config_layer('eos_id')
class EosIdLayer(LayerBase):
    def __init__(self, name, inputs, eos_id, device=None):
        super(EosIdLayer, self).__init__(
            name, 'eos_id', 0, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 1, 'EosIdLayer must have 1 input')
        self.set_layer_size(2)  # boolean output
        self.config.eos_id = eos_id


@config_layer('seqlastins')
class SequenceLastInstanceLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 active_type='linear',
                 trans_type='non-seq',
                 device=None,
                 bias=False):
        super(SequenceLastInstanceLayer, self).__init__(
            name,
            'seqlastins',
            0,
            inputs=inputs,
            device=device,
            active_type=active_type)
        config_assert(
            len(inputs) == 1, 'SequenceLastInstanceLayer must have 1 input')
        self.config.trans_type = trans_type
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('seqfirstins')
class SequenceFirstInstanceLayer(SequenceLastInstanceLayer):
    def __init__(
            self,
            name,
            inputs,
            active_type='linear',
            trans_type='non-seq',
            device=None,
            bias=False, ):
        super(SequenceFirstInstanceLayer, self).__init__(
            name,
            inputs=inputs,
            active_type=active_type,
            device=device,
            bias=bias)
        self.config.trans_type = trans_type
        self.config.select_first = True


@config_layer('seqconcat')
class SequenceConcatLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 active_type='linear',
                 device=None,
                 bias=False):
        super(SequenceConcatLayer, self).__init__(
            name,
            'seqconcat',
            0,
            inputs=inputs,
            device=device,
            active_type=active_type)
        config_assert(
            len(inputs) == 2, 'SequenceConcatLayer must have 2 inputs')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('seqreshape')
class SequenceReshapeLayer(LayerBase):
    def __init__(self,
                 name,
                 size,
                 inputs,
                 active_type='linear',
                 device=None,
                 bias=False):
        super(SequenceReshapeLayer, self).__init__(
            name,
            'seqreshape',
            size,
            inputs=inputs,
            device=device,
            active_type=active_type)
        config_assert(
            len(inputs) == 1, 'SequenceReshapeLayer must have 1 inputs')
        self.set_layer_size(size)
        self.create_bias_parameter(bias, size)


@config_layer('subseq')
class SubSequenceLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 active_type='linear',
                 device=None,
                 bias=False):
        super(SubSequenceLayer, self).__init__(
            name,
            'subseq',
            0,
            inputs=inputs,
            device=device,
            active_type=active_type)
        config_assert(len(inputs) == 3, 'SubSequenceLayer must have 3 inputs')
        input_layer0 = self.get_input_layer(0)
        size = input_layer0.size
        self.set_layer_size(size)
        self.create_bias_parameter(bias, size)


@config_layer('out_prod')
class OuterProdLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(OuterProdLayer, self).__init__(
            name, 'out_prod', 0, inputs=inputs, device=device)
        config_assert(len(inputs) == 2, 'OuterProdLayer must have 2 inputs')
        input_layer0 = self.get_input_layer(0)
        input_layer1 = self.get_input_layer(1)
        self.set_layer_size(input_layer0.size * input_layer1.size)


@config_layer('power')
class PowerLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(PowerLayer, self).__init__(
            name, 'power', 0, inputs=inputs, device=device)
        config_assert(len(inputs) == 2, 'PowerLayer must have 2 inputs')
        input_layer1 = self.get_input_layer(1)
        self.set_layer_size(input_layer1.size)
        input_layer0 = self.get_input_layer(0)
        config_assert(1 == input_layer0.size,
                      'The left input is the exponent and should be of size 1')


@config_layer('slope_intercept')
class SlopeInterceptLayer(LayerBase):
    def __init__(self, name, inputs, slope=1.0, intercept=0.0, device=None):
        super(SlopeInterceptLayer, self).__init__(
            name, 'slope_intercept', 0, inputs=inputs, device=device)
        self.config.slope = slope
        self.config.intercept = intercept
        config_assert(len(inputs) == 1, 'SlopeInterceptLayer must have 1 input')
        input_layer0 = self.get_input_layer(0)
        self.set_layer_size(input_layer0.size)


@config_layer('scaling')
class ScalingLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(ScalingLayer, self).__init__(
            name, 'scaling', 0, inputs=inputs, device=device)
        config_assert(len(inputs) == 2, 'ScalingLayer must have 2 inputs')
        input_layer1 = self.get_input_layer(1)
        self.set_layer_size(input_layer1.size)
        input_layer0 = self.get_input_layer(0)
        config_assert(1 == input_layer0.size,
                      'The left input should be of size 1')


@config_layer('conv_shift')
class ConvShiftLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(ConvShiftLayer, self).__init__(
            name, 'conv_shift', 0, inputs=inputs, device=device)
        config_assert(len(inputs) == 2, 'ConvShiftLayer must have 2 inputs')
        input_layer0 = self.get_input_layer(0)
        self.set_layer_size(input_layer0.size)


@config_layer('convex_comb')
class ConvexCombinationLayer(LayerBase):
    def __init__(self, name, size, inputs, device=None):
        super(ConvexCombinationLayer, self).__init__(
            name, 'convex_comb', size, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 2, 'ConvexCombinationLayer must have 2 inputs')
        config_assert(
            size * self.get_input_layer(0).size == self.get_input_layer(1).size,
            'Wrong input size for ConvexCombinationLayer')
        self.set_layer_size(size)


@config_layer('interpolation')
class InterpolationLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(InterpolationLayer, self).__init__(
            name, 'interpolation', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 3, 'InterpolationLayer must have 3 inputs')
        input_layer0 = self.get_input_layer(0)
        input_layer1 = self.get_input_layer(1)
        input_layer2 = self.get_input_layer(2)
        self.set_layer_size(input_layer1.size)
        config_assert(input_layer0.size == 1, 'weight should be of size 1')
        config_assert(input_layer1.size == input_layer2.size,
                      'the two vector inputs should be of the same size')


@config_layer('bilinear_interp')
class BilinearInterpLayer(LayerBase):
    def __init__(self, name, inputs, **xargs):
        super(BilinearInterpLayer, self).__init__(
            name, 'bilinear_interp', 0, inputs=inputs, **xargs)
        input_layer = self.get_input_layer(0)
        conf = self.config.inputs[0].bilinear_interp_conf
        parse_bilinear(self.inputs[0].bilinear_interp, input_layer.name, conf)
        self.set_cnn_layer(name, conf.out_size_y, conf.out_size_x,
                           conf.image_conf.channels)


@config_layer('sum_to_one_norm')
class SumToOneNormLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(SumToOneNormLayer, self).__init__(
            name, 'sum_to_one_norm', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 1, 'SumToOneNormLayer must have 1 input')
        input_layer0 = self.get_input_layer(0)
        self.set_layer_size(input_layer0.size)


@config_layer('cos_vm')
class CosSimVecMatLayer(LayerBase):
    def __init__(self, name, size, inputs, cos_scale=1.0, device=None):
        super(CosSimVecMatLayer, self).__init__(
            name, 'cos_vm', size, inputs=inputs, device=device)
        self.config.cos_scale = cos_scale
        config_assert(
            len(self.inputs) == 2, 'CosSimVecMatLayer must have 2 inputs')
        config_assert(
            size * self.get_input_layer(0).size == self.get_input_layer(1).size,
            'Wrong input size for CosSimVecMatLayer')


@config_layer('sampling_id')
class SamplingIdLayer(LayerBase):
    def __init__(self, name, inputs, device=None):
        super(SamplingIdLayer, self).__init__(
            name, 'sampling_id', 0, inputs=inputs, device=device)
        config_assert(
            len(self.inputs) == 1, 'SamplingIdLayer must have 1 input')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)


# AverageLayer: "average" for each sample within a sequence.
# average_stratrgy: set to one of the following:
# 'average': plain average.
# 'sum': sum each sample instead of average (which is divide by sample_num).
# 'squarerootn': sum each sample, but divide by sqrt(sample_num).
@config_layer('average')
class AverageLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 average_strategy='average',
                 trans_type='non-seq',
                 active_type='linear',
                 device=None,
                 bias=False):
        super(AverageLayer, self).__init__(
            name,
            'average',
            0,
            inputs=inputs,
            device=device,
            active_type=active_type)
        self.config.average_strategy = average_strategy
        self.config.trans_type = trans_type
        config_assert(len(inputs) == 1, 'AverageLayer must have 1 input')
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            self.set_layer_size(input_layer.size)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('cos')
class CosSimLayer(LayerBase):
    def __init__(self, name, inputs, cos_scale=5, device=None):
        super(CosSimLayer, self).__init__(
            name, 'cos', 1, inputs=inputs, device=device)
        config_assert(len(self.inputs) == 2, 'CosSimLayer must have 2 inputs')
        config_assert(
            self.get_input_layer(0).size == self.get_input_layer(1).size,
            'inputs of CosSimLayer must have same dim')
        self.config.cos_scale = cos_scale


@config_layer('tensor')
class TensorLayer(LayerBase):
    def __init__(self, name, size, inputs, device=None, bias=True, **xargs):
        super(TensorLayer, self).__init__(
            name, 'tensor', size, inputs=inputs, device=device, **xargs)
        config_assert(len(self.inputs) == 2, 'TensorLayer must have 2 inputs')
        config_assert(size > 0, 'size must be positive')
        config_assert(inputs[1].parameter_name == None,
                      'second parameter should be None.')
        input_layer0 = self.get_input_layer(0)
        input_layer1 = self.get_input_layer(1)
        psize = size * input_layer0.size * input_layer1.size
        dims = [input_layer0.size, input_layer1.size, size]
        self.create_input_parameter(0, psize, dims)
        self.create_bias_parameter(bias, size)


@config_layer('mixed')
class MixedLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 size=0,
                 bias=True,
                 error_clipping_threshold=None,
                 **xargs):
        config_assert(inputs, 'inputs cannot be empty')
        super(MixedLayer, self).__init__(
            name, 'mixed', size, inputs=inputs, **xargs)
        operator_input_index = []
        for operator in self.operators:
            operator_conf = operator.operator_conf
            for i in xrange(1, len(operator.input_layer_names)):
                input_index = len(self.config.inputs)
                operator_conf.input_indices.append(input_index)
                input_config = Input(operator.input_layer_names[i])
                self.inputs.append(input_config)
                layer_input = self.config.inputs.add()
                layer_input.input_layer_name = input_config.input_layer_name
            for input_index in operator_conf.input_indices:
                input_layer = self.get_input_layer(input_index)
                operator_conf.input_sizes.append(input_layer.size)
                operator_input_index.append(input_index)
            if self.config.size == 0:
                size = operator.calc_output_size(operator_conf.input_sizes)
                if size != 0:
                    self.set_layer_size(size)
            else:
                sz = operator.calc_output_size(operator_conf.input_sizes)
                if sz != 0:
                    config_assert(
                        sz == self.config.size,
                        "different inputs have different size: %s vs. %s" %
                        (sz, self.config.size))
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            input = self.inputs[input_index]
            if input_index not in operator_input_index:
                config_assert(
                    isinstance(input, Projection),
                    "input should be projection or operation")
            if self.config.size == 0 and isinstance(input, Projection):
                size = input.calc_output_size(input_layer)
                if size != 0:
                    self.set_layer_size(size)
            elif isinstance(input, Projection):
                sz = input.calc_output_size(input_layer)
                if sz != 0:
                    config_assert(
                        sz == self.config.size,
                        "different inputs have different size: %s vs. %s" %
                        (sz, self.config.size))
        config_assert(size != 0, "size is not set")

        for input_index in xrange(len(self.inputs)):
            input = self.inputs[input_index]
            if isinstance(input, Projection):
                input_layer = self.get_input_layer(input_index)
                input.proj_conf.input_size = input_layer.size
                input.proj_conf.output_size = size

                input_config = self.config.inputs[input_index]
                input_config.proj_conf.CopyFrom(input.proj_conf)
                input_config.proj_conf.name = gen_parameter_name(name,
                                                                 input_index)
                psize = input.calc_parameter_size(input_layer.size, size)
                dims = input.calc_parameter_dims(input_layer.size, size)
                self.create_input_parameter(input_index, psize, dims)

        for operator in self.operators:
            operator_conf = operator.operator_conf
            operator_conf.output_size = self.config.size
            operator.check_dims()
            record_operator_conf = self.config.operator_confs.add()
            record_operator_conf.CopyFrom(operator_conf)

        psize = self.config.size
        if isinstance(self.inputs[0], ConvProjection):
            self.config.shared_biases = True
            psize = 0
            for input in self.inputs:
                psize += input.calc_bias_size()

        if bias:
            self.config.bias_size = psize
            self.create_bias_parameter(bias, psize)

        if error_clipping_threshold is not None:
            self.config.error_clipping_threshold = error_clipping_threshold


# like MixedLayer, but no bias parameter
@config_func
def ExpressionLayer(name, inputs, **xargs):
    MixedLayer(name, inputs, bias=False, **xargs)


@config_layer('concat')
class ConcatenateLayer(LayerBase):
    def __init__(self, name, inputs, bias=False, **xargs):
        config_assert(inputs, 'inputs cannot be empty')
        config_assert(not bias, 'ConcatenateLayer cannot support bias.')
        super(ConcatenateLayer, self).__init__(
            name, 'concat', 0, inputs=inputs, **xargs)
        size = 0
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            input = self.inputs[input_index]
            if self.config.size == 0:
                size += input_layer.size

        self.set_layer_size(size)


# like concat layer, but each input layer was processed by a Projection.
@config_layer('concat2')
class ConcatenateLayer2(LayerBase):
    def __init__(self, name, inputs, bias=False, **xargs):
        config_assert(inputs, 'inputs cannot be empty')
        super(ConcatenateLayer2, self).__init__(
            name, 'concat2', 0, inputs=inputs, **xargs)

        if isinstance(self.inputs[0], ConvProjection):
            for input_index in xrange(len(self.inputs) - 1):
                input = self.inputs[input_index + 1]
                config_assert(
                    isinstance(input, ConvProjection),
                    "The first input of ConcatenateLayer2 is ConvProjection, "
                    "the other inputs should also be ConvProjection.")

        size = 0
        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            input = self.inputs[input_index]
            output_size = input.calc_output_size(input_layer)
            config_assert(output_size != 0, "proj output size is not set")
            size += output_size

        self.set_layer_size(size)

        for input_index in xrange(len(self.inputs)):
            input_layer = self.get_input_layer(input_index)
            input = self.inputs[input_index]
            input.proj_conf.input_size = input_layer.size
            input.proj_conf.output_size = input.calc_output_size(input_layer)

            input_config = self.config.inputs[input_index]
            input_config.proj_conf.CopyFrom(input.proj_conf)
            input_config.proj_conf.name = gen_parameter_name(name, input_index)
            psize = input.calc_parameter_size(input.proj_conf.input_size,
                                              input.proj_conf.output_size)
            dims = input.calc_parameter_dims(input.proj_conf.input_size,
                                             input.proj_conf.output_size)
            self.create_input_parameter(input_index, psize, dims)

        psize = self.config.size
        if isinstance(self.inputs[0], ConvProjection):
            self.config.shared_biases = True
            psize = 0
            for input in self.inputs:
                psize += input.calc_bias_size()

        if bias:
            self.config.bias_size = psize
            self.create_bias_parameter(bias, psize)


@config_layer('recurrent')
class RecurrentLayer(LayerBase):
    def __init__(self, name, inputs, reversed=False, bias=True, **xargs):
        super(RecurrentLayer, self).__init__(name, 'recurrent', 0, inputs,
                                             **xargs)
        config_assert(len(self.inputs) == 1, 'RecurrentLayer must have 1 input')
        input_layer = self.get_input_layer(0)
        size = input_layer.size
        self.set_layer_size(size)
        self.config.reversed = reversed
        dims = [size, size]
        self.create_input_parameter(0, size * size, dims)
        self.create_bias_parameter(bias, self.config.size)


@config_layer('lstmemory')
class LstmLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 reversed=False,
                 active_gate_type="sigmoid",
                 active_state_type="sigmoid",
                 bias=True,
                 **xargs):
        super(LstmLayer, self).__init__(name, 'lstmemory', 0, inputs, **xargs)
        config_assert(len(self.inputs) == 1, 'LstmLayer must have 1 input')
        input_layer = self.get_input_layer(0)
        #check input_layer.size is divided by 4
        config_assert(input_layer.size % 4 == 0, "size % 4 should be 0!")
        size = input_layer.size / 4
        self.set_layer_size(size)
        self.config.reversed = reversed
        self.config.active_gate_type = active_gate_type
        self.config.active_state_type = active_state_type
        self.create_input_parameter(0, size * size * 4, [size, size, 4])
        #bias includes 3 kinds of peephole, 4 + 3 = 7
        self.create_bias_parameter(bias, size * 7)


@config_layer('lstm_step')
class LstmStepLayer(LayerBase):
    def __init__(self,
                 name,
                 size,
                 inputs,
                 active_gate_type="sigmoid",
                 active_state_type="sigmoid",
                 bias=True,
                 **xargs):
        super(LstmStepLayer, self).__init__(name, 'lstm_step', size, inputs,
                                            **xargs)
        config_assert(len(inputs) == 2, 'LstmStepLayer must have 2 inputs')
        input_layer0 = self.get_input_layer(0)
        input_layer1 = self.get_input_layer(1)
        config_assert(input_layer0.size == 4 * size,
                      'input_layer0.size != 4 * layer.size')
        config_assert(input_layer1.size == size,
                      'input_layer1.size != layer.size')
        self.config.active_gate_type = active_gate_type
        self.config.active_state_type = active_state_type
        self.create_bias_parameter(bias, size * 3)


# get the specific output from the input layer.
@config_layer('get_output')
class GetOutputLayer(LayerBase):
    def __init__(self, name, size, inputs):
        super(GetOutputLayer, self).__init__(name, 'get_output', size, inputs)
        config_assert(
            len(self.inputs) == 1, 'GetOutputLayer must have 1 inputs')
        inputs = self.inputs[0]
        config_assert(inputs.input_layer_argument,
                      'input_layer_argument cannot be empty')


@config_layer('mdlstmemory')
class MDLstmLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 directions=True,
                 active_gate_type="sigmoid",
                 active_state_type="sigmoid",
                 bias=True,
                 **xargs):
        super(MDLstmLayer, self).__init__(name, 'mdlstmemory', 0, inputs,
                                          **xargs)
        config_assert(len(self.inputs) == 1, 'MDLstmLayer must have 1 input')
        input_layer = self.get_input_layer(0)
        dim_num = len(directions)
        #check input_layer.size is divided by (3+dim_num)
        config_assert(input_layer.size % (3 + dim_num) == 0,
                      "size % (dim_num) should be 0!")
        size = input_layer.size / (3 + dim_num)
        self.set_layer_size(size)
        self.config.active_gate_type = active_gate_type
        self.config.active_state_type = active_state_type
        for i in xrange(len(directions)):
            self.config.directions.append(int(directions[i]))
        self.create_input_parameter(0, size * size * (3 + dim_num),
                                    [size, size, 3 + dim_num])
        #bias includes 3 kinds of peephole, 3+dim_num+2+dim_num
        self.create_bias_parameter(bias, size * (5 + 2 * dim_num))


@config_layer('gated_recurrent')
class GatedRecurrentLayer(LayerBase):
    def __init__(self,
                 name,
                 inputs,
                 reversed=False,
                 active_gate_type="sigmoid",
                 bias=True,
                 **xargs):
        super(GatedRecurrentLayer, self).__init__(name, 'gated_recurrent', 0,
                                                  inputs, **xargs)
        config_assert(
            len(self.inputs) == 1, 'GatedRecurrentLayer must have 1 input')
        input_layer = self.get_input_layer(0)
        #check input_layer.size is divided by 3
        config_assert(input_layer.size % 3 == 0, "size % 3 should be 0!")
        size = input_layer.size / 3
        self.set_layer_size(size)
        self.config.reversed = reversed
        self.config.active_gate_type = active_gate_type
        self.create_input_parameter(0, size * size * 3, [size, size * 3])
        self.create_bias_parameter(bias, size * 3)


@config_layer('gru_step')
class GruStepLayer(LayerBase):
    def __init__(self,
                 name,
                 size,
                 inputs,
                 active_gate_type="sigmoid",
                 bias=True,
                 **xargs):
        super(GruStepLayer, self).__init__(name, 'gru_step', size, inputs,
                                           **xargs)
        config_assert(len(self.inputs) == 2, 'GruStepLayer must have 2 input')
        input_layer0 = self.get_input_layer(0)
        input_layer1 = self.get_input_layer(1)
        config_assert(input_layer0.size == 3 * size,
                      'input_layer0.size != 3 * layer.size')
        config_assert(input_layer1.size == size,
                      'input_layer1.size != layer.size')
        self.config.active_gate_type = active_gate_type
        self.create_input_parameter(0, size * size * 3, [size, size * 3])
        self.create_bias_parameter(bias, size * 3)


'''
 A layer for calculating the cost of sequential conditional random field model.
 Example: CRFLayer(name="crf_cost", size=label_num,
                   inputs=["output", "label", "weight"])
          where "weight" is optional, one weight for each sequence
 @param coeff: weight of the layer
'''


@config_layer('crf')
class CRFLayer(LayerBase):
    def __init__(self, name, size, inputs, coeff=1.0, device=None):
        super(CRFLayer, self).__init__(name, 'crf', size, inputs, device=device)
        config_assert(2 <= len(self.inputs) <= 3,
                      'CRFLayer must have 2 or 3 inputs')
        self.create_input_parameter(0, size * (size + 2), [size, size + 2])
        self.config.coeff = coeff


'''
 A layer for calculating the decoding sequence of sequential conditional
 random field model.
 The decoding sequence is stored in output_.ids
 If a second input is provided, it is treated as the ground-truth label, and
 this layer will also calculate error, output_.value[i] is 1 for incorrect
 decoding or 0 for correct decoding
'''


@config_layer('crf_decoding')
class CRFDecodingLayer(LayerBase):
    def __init__(self, name, size, inputs, device=None):
        super(CRFDecodingLayer, self).__init__(
            name, 'crf_decoding', size, inputs, device=device)
        config_assert(
            len(self.inputs) <= 2,
            'CRFDecodingLayer cannot have more than 2 inputs')
        self.create_input_parameter(0, size * (size + 2), [size, size + 2])


@config_layer('ctc')
class CTCLayer(LayerBase):
    def __init__(self, name, size, inputs, norm_by_times=False, device=None):
        super(CTCLayer, self).__init__(name, 'ctc', size, inputs, device=device)
        self.config.norm_by_times = norm_by_times
        config_assert(len(self.inputs) == 2, 'CTCLayer must have 2 inputs')


@config_layer('warp_ctc')
class WarpCTCLayer(LayerBase):
    def __init__(self,
                 name,
                 size,
                 inputs,
                 blank=0,
                 norm_by_times=False,
                 device=None):
        super(WarpCTCLayer, self).__init__(
            name, 'warp_ctc', size=size, inputs=inputs, device=device)
        self.config.blank = blank
        self.config.norm_by_times = norm_by_times
        config_assert(len(self.inputs) == 2, 'WarpCTCLayer must have 2 inputs')
        input_layer = self.get_input_layer(0)
        config_assert(
            (input_layer.active_type == '' or
             input_layer.active_type == 'linear'),
            "Expecting the active_type of input layer to be linear or null")


@config_layer('recurrent_layer_group')
class RecurrentLayerGroup(LayerBase):
    def __init__(self, name, device=None):
        global g_pass_height_width
        g_pass_height_width = False
        super(RecurrentLayerGroup, self).__init__(
            name, 'recurrent_layer_group', 0, inputs=[], device=device)


# Deprecated, use a new layer specific class instead
@config_func
def Layer(name, type, **xargs):
    layers = {}
    layers.update(g_cost_map)
    layers.update(g_layer_type_map)
    layer_func = layers.get(type)
    config_assert(layer_func, "layer type '%s' not supported." % type)
    return layer_func(name, **xargs)


@config_func
def ParameterHook(type, **kwargs):
    if type == 'pruning':
        mask_filename = kwargs.get('mask_filename', None)
        assert mask_filename is not None
        hook = ParameterUpdaterHookConfig()
        hook.type = type
        hook.purning_mask_filename = mask_filename
        return hook
    else:
        return None


@config_func
def Parameter(name,
              size,
              device,
              dims,
              learning_rate=None,
              momentum=None,
              decay_rate=None,
              decay_rate_l1=None,
              initial_mean=None,
              initial_std=None,
              initial_strategy=None,
              initial_smart=None,
              num_batches_regularization=None,
              sparse_remote_update=None,
              sparse_update=None,
              gradient_clipping_threshold=None,
              sparse=None,
              format=None,
              need_compact=None,
              is_static=None,
              is_shared=None,
              update_hooks=None):

    config_assert(name not in g_parameter_map,
                  'Duplicated parameter name: ' + name)

    para = g_config.model_config.parameters.add()
    para.name = name
    para.size = size
    if device is not None:
        para.device = int(device)
    para.dims.extend(dims)

    if learning_rate is not None:
        para.learning_rate = float(learning_rate)

    momentum = default(momentum, g_default_momentum)
    if momentum is not None:
        para.momentum = float(momentum)

    config_assert(not momentum or not decay_rate_l1,
                  "momentum and decay_rate_l1 cannot both be non-zero")

    decay_rate = default(decay_rate, g_default_decay_rate)
    if decay_rate is not None:
        para.decay_rate = decay_rate

    if decay_rate_l1 is not None:
        para.decay_rate_l1 = decay_rate_l1
    para.initial_std = default(initial_std, g_default_initial_std)
    para.initial_mean = default(initial_mean, g_default_initial_mean)

    num_batches_regularization = default(num_batches_regularization,
                                         g_default_num_batches_regularization)
    if num_batches_regularization is not None:
        para.num_batches_regularization = int(num_batches_regularization)

    if sparse_remote_update is not None:
        para.sparse_remote_update = sparse_remote_update
        if sparse_remote_update:
            g_config.opt_config.use_sparse_remote_updater = True
    if sparse_update is not None:
        para.sparse_update = sparse_update
    gradient_clipping_threshold = default(gradient_clipping_threshold,
                                          g_default_gradient_clipping_threshold)
    if gradient_clipping_threshold is not None:
        para.gradient_clipping_threshold = gradient_clipping_threshold
    para.initial_strategy = default(initial_strategy,
                                    g_default_initial_strategy)
    para.initial_smart = default(initial_smart, g_default_initial_smart)
    if para.initial_smart:
        para.initial_mean = 0.
        if len(para.dims) != 0:
            para.initial_std = 1. / math.sqrt(para.dims[0])
        else:
            print(
                "Use initial_smart, but dims not set. Initial_smart may not be used in this layer"
            )
            traceback.print_exc()
            para.initial_std = 1. / math.sqrt(para.size)
    if g_default_compact_func is not None:
        sparse, format, need_compact = g_default_compact_func(para.name)

    if sparse is not None:
        para.is_sparse = sparse
    if format is not None:
        para.format = format
    if need_compact is not None:
        para.need_compact = need_compact
    if is_static is not None:
        para.is_static = is_static
    config_assert(not para.sparse_remote_update or not para.is_static,
                  "sparse_remote_update and is_static cannot both be true")
    if is_shared is not None:
        para.is_shared = is_shared

    update_hooks = default(update_hooks, g_default_update_hooks)

    if update_hooks is not None:
        if hasattr(update_hooks, '__call__'):
            update_hooks = update_hooks(para.name)

        if isinstance(update_hooks, list):
            for hook in update_hooks:
                para.update_hooks.extend([hook])
        else:
            para.update_hooks.extend(update_hooks)

    g_parameter_map[name] = para


@config_func
def default_initial_std(val):
    global g_default_initial_std
    g_default_initial_std = val


@config_func
def default_initial_mean(val):
    global g_default_initial_mean
    g_default_initial_mean = val


@config_func
def default_initial_strategy(val):
    global g_default_initial_strategy
    g_default_initial_strategy = val


@config_func
def default_initial_smart(val):
    global g_default_initial_smart
    g_default_initial_smart = val


@config_func
def default_momentum(val):
    global g_default_momentum
    g_default_momentum = val


@config_func
def default_decay_rate(val):
    global g_default_decay_rate
    g_default_decay_rate = val


@config_func
def default_num_batches_regularization(val):
    global g_default_num_batches_regularization
    g_default_num_batches_regularization = val


@config_func
def default_gradient_clipping_threshold(val):
    global g_default_gradient_clipping_threshold
    g_default_gradient_clipping_threshold = val


@config_func
def default_device(val):
    global g_default_device
    g_default_device = val


@config_func
def default_update_hooks(val):
    global g_default_update_hooks
    g_default_update_hooks = val


@config_func
def default_compact_func(val):
    global g_default_compact_func
    g_default_compact_func = val


def make_importer(config_dir, config_args):
    def Import(config_file, local_args={}):
        if not config_file.startswith('/'):
            config_file = config_dir + '/' + config_file
            g_config.config_files.append(config_file)
        execfile(config_file,
                 make_config_environment(config_file, config_args), local_args)

    return Import


settings = dict(
    batch_size=None,
    mini_batch_size=None,
    algorithm='async_sgd',
    async_lagged_grad_discard_ratio=1.5,
    learning_method='momentum',
    num_batches_per_send_parameter=None,
    num_batches_per_get_parameter=None,
    center_parameter_update_method=None,
    learning_rate=1.,
    learning_rate_decay_a=0.,
    learning_rate_decay_b=0.,
    learning_rate_schedule='poly',
    learning_rate_args='',
    l1weight=0.1,
    l2weight=0.,
    l2weight_zero_iter=0,
    c1=0.0001,
    backoff=0.5,
    owlqn_steps=10,
    max_backoff=5,
    average_window=0,
    do_average_in_cpu=False,
    max_average_window=None,
    ada_epsilon=1e-6,
    ada_rou=0.95,
    delta_add_rate=1.0,
    shrink_parameter_value=0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8, )

settings_deprecated = dict(usage_ratio=1., )

trainer_settings = dict(
    save_dir="./output/model",
    init_model_path=None,
    start_pass=0, )


@config_func
def Settings(**args):
    for k, v in args.iteritems():
        if k == "usage_ratio":
            logger.warning(
                "Deprecated: define usage_ratio in DataConfig instead")
            if g_config.HasField("data_config"):
                g_config.data_config.__setattr__(k, v)
            settings_deprecated[k] = v
            continue
        elif k in settings:
            settings[k] = v
        elif k in trainer_settings:
            trainer_settings[k] = v
        else:
            logger.fatal('Unkown setting: %s' % k)


@config_func
def cluster_config(**args):
    pass


@config_func
def EnableSubmodelSuffix(flag=True):
    """
    If enabled, the layer and evaluator names in submodel will be automatically
    appended with @submodel_name
    """
    global g_add_submodel_suffix
    g_add_submodel_suffix = flag


def make_config_environment(config_file, config_args):
    def make_setter(k):
        def setter(v):
            logger.fatal("Obsolete: use Settings(%s=%s, ...) instead" % (k, v))

        return setter

    funcs = {}
    funcs.update(g_config_funcs)

    for k in settings.iterkeys():
        funcs[k] = make_setter(k)
    for k in settings_deprecated.iterkeys():
        funcs[k] = make_setter(k)
    config_dir = os.path.dirname(config_file)
    if not config_dir:
        config_dir = '.'

    funcs.update(
        Import=make_importer(config_dir, config_args),
        get_config_arg=make_get_config_arg(config_args), )

    funcs.update(g_extended_config_funcs)

    return funcs


def make_get_config_arg(config_args):
    def get_config_arg(name, type, default=None):
        if type == bool:
            s = config_args.get(name)
            if not s:
                return default
            if s == 'True' or s == '1' or s == 'true':
                return True
            if s == 'False' or s == '0' or s == 'false':
                return False
            raise ValueError('Value of config_arg %s is not boolean' % name)
        else:
            return type(config_args.get(name, default))

    return get_config_arg


def importlib(name):
    __import__(name)
    return sys.modules[name]


def find_caller():
    stack = traceback.extract_stack()
    for s in stack[-4::-1]:
        if not s[0].endswith('config_parser.py'):
            return s[0], s[1], s[2]
    return "(unknown file)", 0, "(unknown function)"


def my_fatal(s):
    logger.critical(s)
    raise Exception()


_parse_config_hooks = set()


def register_parse_config_hook(f):
    """
    Register a hook function for parse_config. parse_config will invoke the hook
    at the beginning of parse. This make it possible to reset global state for
    for constructing the model.
    """
    _parse_config_hooks.add(f)


def parse_config(config_file, config_arg_str):
    '''
    @param config_arg_str: a string of the form var1=val1,var2=val2. It will be
    passed to config script as a dictionary CONFIG_ARGS
    '''
    init_config_environment()
    for hook in _parse_config_hooks:
        hook()

    config_args = {}

    logger.findCaller = find_caller
    logger.fatal = my_fatal

    g_config.model_config.type = "nn"
    if config_arg_str:
        config_args = dict([f.split('=') for f in config_arg_str.split(',')])

    global g_command_config_args
    g_command_config_args.update(config_args)

    extension_module_name = config_args.get('extension_module_name')
    if extension_module_name:
        global g_extended_config_funcs
        extension_module = importlib(extension_module_name)
        g_extended_config_funcs = extension_module.get_config_funcs(g_config)

    g_config.model_config.type = 'nn'

    global g_current_submodel, g_root_submodel
    g_root_submodel = g_config.model_config.sub_models.add()
    g_root_submodel.name = 'root'
    g_root_submodel.is_recurrent_layer_group = False
    g_current_submodel = g_root_submodel

    # for paddle on spark, need support non-file config.
    # you can use parse_config like below:
    #
    # from paddle.trainer.config_parser import parse_config
    # def configs():
    #    #your paddle config code, which is same as config file.
    #
    # config = parse_config(configs, "is_predict=1")
    # # then you get config proto object.
    if hasattr(config_file, '__call__'):
        config_file.func_globals.update(
            make_config_environment("", config_args))
        config_file()
    else:
        execfile(config_file, make_config_environment(config_file, config_args))
    for k, v in settings.iteritems():
        if v is None:
            continue
        g_config.opt_config.__setattr__(k, v)

    for k, v in trainer_settings.iteritems():
        if v is None:
            continue
        g_config.__setattr__(k, v)

    for name in g_config.model_config.input_layer_names:
        assert name in g_layer_map, \
            'input name "%s" does not correspond to a layer name' % name
        assert (g_layer_map[name].type == "data" or g_layer_map[name].type == "data_trim"), \
            'The type of input layer "%s" is not "data"' % name
    for name in g_config.model_config.output_layer_names:
        assert name in g_layer_map, \
            'input name "%s" does not correspond to a layer name' % name
    return g_config


def parse_config_and_serialize(config_file, config_arg_str):
    try:
        config = parse_config(config_file, config_arg_str)
        #logger.info(config)
        return config.SerializeToString()
    except:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    try:
        config = parse_config(sys.argv[1], '')
        config.SerializeToString()
        __real_print__(str(config))
    except:
        traceback.print_exc()
        raise
