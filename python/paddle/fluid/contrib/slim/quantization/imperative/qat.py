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

import collections
import logging
import numpy as np
import sys
import os
import warnings

import paddle
from paddle.fluid import dygraph, core, framework, unique_name
from paddle.fluid.executor import Executor
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX
from paddle.nn import Linear, Conv2D, Conv2DTranspose, MaxPool2D, MaxPool1D
from paddle.nn import BatchNorm1D, BatchNorm2D, BatchNorm3D, SyncBatchNorm
from paddle.fluid.dygraph.nn import BatchNorm, Pool2D
from paddle.fluid.io import load_inference_model, save_inference_model
from paddle.nn.layer.activation import ReLU, LeakyReLU, Sigmoid, ReLU6
from paddle.nn.layer.activation import Tanh, Softmax, PReLU, Swish
from paddle.fluid.log_helper import get_logger
from . import quant_nn
from .. import quantization_pass
from . import utils

__all__ = ['ImperativeQuantAware']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeQuantAware(object):
    """
    Applying quantization aware training (QAT) to dgraph model.
    """

    def __init__(self,
                 quantizable_layer_type=['Conv2D', 'Linear'],
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_preprocess_layer=None,
                 act_preprocess_layer=None,
                 weight_quantize_layer=None,
                 act_quantize_layer=None):
        """
        The constructor for ImperativeQuantAware.

        Args:
            quantizable_layer_type(list[str]): List the type of layers that
                will be quantized. Default is ['Conv2D', 'Linear'].
                The quantizable_op_type in QuantizationFreezePass and
                ConvertToInt8Pass must be the same as this.
            weight_quantize_type(str): quantization type for weights,
                which supports 'abs_max' now. The 'moving_average_abs_max'
                usually is not used for weights, since weights are fixed
                once the model is well trained.
            activation_quantize_type(str): quantization type for activations,
                which supports 'abs_max' and 'moving_average_abs_max' now.
                If using 'abs_max' mode, the quantization scale will be
                calculated dynamically each step in both training and testing
                period. If using 'moving_average_abs_max', the static
                quantization scale will be calculated during training and
                used in inference.
            weight_bits(int): quantization bit number for weights,
                whereas the bias is not quantized.
            activation_bits(int): quantization bit number for activations.
            moving_rate(float): the parameter for 'moving_average_abs_max'
                quantization.
            weight_preprocess_layer(paddle.nn.Layer, optional): A paddle
                Layer that defines how to preprocess weight before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized weight and function returns
                processed weight to be quantized.
                If None, the weight will be quantized directly.
                Default is None.
            act_preprocess_layer(paddle.nn.Layer, optional): A paddle Layer
                that defines how to preprocess activation before quantization.
                Using this can quickly test if user's preprocess method works
                or not. The input is non-quantized activation and function returns
                processed activation to be quantized.
                If None, the activation will be quantized directly.
                Default is None.
            weight_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that
                defines how to quantize weight.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                weight and returns dequantized weight.
                If None, will use uantization op defined by 'weight_quantize_type'.
                Default is None.
            act_quantize_layer(paddle.nn.Layer, optional): A paddle Layer that defines
                how to quantize activation.
                Using this can quickly test if user's quantization method works or not.
                In this layer, user should both define quantization method and
                dequantization method, that is, the function's input is non-quantized
                activation and returns dequantized activation. 
                If None, will use quantization op defined by 'activation_quantize_type'.
                Default is None.

        Note:
            If user sets attribute 'skip_quant' to a Layer that support dynamic
            quantization and sets it to true, the layer would not be quantized
            during training. If this attribute is not sets or the attribute is
            false, the Layer would be qunatized in training.

        Examples 1:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware
            from paddle.vision.models \
                import resnet
            
            model = resnet.resnet50(pretrained=True)

            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')
            
            # Add the fake quant logical.
            # The original model will be rewrite.
            # The outscale of outputs in supportted layers would be calculated.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...
            
            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./resnet50_qat",
                input_spec=[
                    paddle.static.InputSpec(
                    shape=[None, 3, 224, 224], dtype='float32')])

        Examples 2:
        .. code-block:: python

            import paddle
            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware

            class ImperativeModel(paddle.nn.Layer):
                def __init__(self):
                    super(ImperativeModel, self).__init__()
                    # self.linear_0 would skip the quantization.
                    self.linear_0 = paddle.nn.Linear(784, 400)
                    self.linear_0.skip_quant = True

                    # self.linear_1 would not skip the quantization.
                    self.linear_1 = paddle.nn.Linear(400, 10)
                    self.linear_1.skip_quant = False

                def forward(self, inputs):
                    x = self.linear_0(inputs)
                    x = self.linear_1(inputs)
                    return x

            model = ImperativeModel()
            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')

            # Add the fake quant logical.
            # The original model will be rewrite.
            #
            # There is only one Layer(self.linear1) would be added the
            # fake quant logical.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...

            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                layer=model,
                model_path="./imperative_model_qat")
        """
        super(ImperativeQuantAware, self).__init__()

        kwargs = {
            "quantizable_layer_type": quantizable_layer_type,
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_preprocess_layer": weight_preprocess_layer,
            "act_preprocess_layer": act_preprocess_layer,
            "weight_quantize_layer": weight_quantize_layer,
            "act_quantize_layer": act_quantize_layer
        }

        self._quantize_inputs = ImperativeQuantizeInputs(**kwargs)

        self._calc_output_scale = ImperativeCalcOutputScale()

    def quantize(self, model):
        """
        According to weights' and activations' quantization types,
        the model will be added some fake quant ops, such as
        fake_quantize_dequantize_moving_average_abs_max,
        fake_quantize_dequantize_abs_max and so on. At the same time,
        the out_scale value of outputs would be calculated.

        Args:
            model(fluid.dygraph.Layer): the model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."
        self._quantize_inputs.apply(model)
        self._calc_output_scale.apply(model)

    def save_quantized_model(self, layer, path, input_spec=None, **config):
        self._calc_output_scale.save_quantized_model(layer, path, input_spec,
                                                     **config)


class ImperativeQuantizeInputs(object):
    """
    Based on the input params, add the quant_dequant computational
    logic both for activation inputs and weight inputs.
    """

    def __init__(self,
                 quantizable_layer_type=['Conv2D', 'Linear'],
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 weight_bits=8,
                 activation_bits=8,
                 moving_rate=0.9,
                 weight_preprocess_layer=None,
                 act_preprocess_layer=None,
                 weight_quantize_layer=None,
                 act_quantize_layer=None):
        """
        The constructor for ImperativeQuantizeInputs. 

        Please refer to the args of ImperativeQuantAware.
        """
        super(ImperativeQuantizeInputs, self).__init__()

        self._quantizable_layer_type = tuple(
            utils._quant_layers_map[layer]
            if layer in utils._quant_layers_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(layer, str), \
                "%s is unspported to be quantized." % layer

        quantize_type = {
            'abs_max', 'moving_average_abs_max', 'channel_wise_abs_max'
        }
        assert weight_quantize_type in quantize_type, \
            "Unsupported weight_quantize_type: %s. It can only " \
            "be abs_max or moving_average_abs_max or " \
            "channel_wise_abs_max." % weight_quantize_type
        assert activation_quantize_type != 'channel_wise_abs_max' \
            and activation_quantize_type in quantize_type, \
            "Unsupported activation_quantize_type: %s. It can " \
            "only be abs_max or moving_average_abs_max now." \
            % activation_quantize_type

        bits_check = lambda bits: isinstance(bits, int) \
            and bits >= 0 and bits <= 16
        assert bits_check(weight_bits), \
            "weight_bits should be 1, 2,... or 16."
        assert bits_check(activation_bits), \
            "activation_bits should be 1, 2,... or 16."

        layer_check = lambda method: method is None or \
            issubclass(method, dygraph.layers.Layer)
        assert layer_check(weight_preprocess_layer), \
            "weight_preprocess should be nn.Layer."
        assert layer_check(act_preprocess_layer), \
            "act_preprocess should be nn.Layer."
        assert layer_check(weight_quantize_layer), \
            "weight_quantize should be nn.Layer."
        assert layer_check(act_quantize_layer), \
            "act_quantize should be nn.Layer."

        self._kwargs = {
            "weight_quantize_type": weight_quantize_type,
            "activation_quantize_type": activation_quantize_type,
            "weight_bits": weight_bits,
            "activation_bits": activation_bits,
            "moving_rate": moving_rate,
            "weight_pre_layer": weight_preprocess_layer,
            "act_pre_layer": act_preprocess_layer,
            "weight_quant_layer": weight_quantize_layer,
            "act_quant_layer": act_quantize_layer
        }

    def apply(self, model):
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."

        for name, layer in model.named_sublayers():
            if not isinstance(layer, self._quantizable_layer_type) \
                or (hasattr(layer, "skip_quant") \
                    and layer.skip_quant == True):
                continue

            # TODO(jc): optimize this module
            last_idx = 0
            idx = 0
            obj = model
            while idx < len(name):
                if (name[idx] == '.'):
                    if hasattr(obj, name[last_idx:idx]):
                        obj = getattr(obj, name[last_idx:idx])
                        last_idx = idx + 1
                idx += 1
            target = name[last_idx:idx]

            quant_layer = self._get_quantized_layer(layer)
            setattr(quant_layer, "layer_name", layer.full_name())
            setattr(obj, target, quant_layer)

    def _get_quantized_layer(self, layer):
        quant_layer_name = None
        for key, value in utils._quant_layers_map.items():
            if isinstance(layer, value):
                quant_layer_name = 'Quantized' + key
                break
        assert quant_layer_name is not None, \
            "The layer %s is unsupported to be quantized." \
            % layer.full_name()

        layer_with_weight = ['QuantizedConv2D', 'QuantizedLinear']
        if quant_layer_name not in layer_with_weight:
            quant_layer_name = 'QuantizedNoweightLayer'

        return quant_nn.__dict__[quant_layer_name](layer, **self._kwargs)


class ImperativeCalcOutputScale(object):
    def __init__(self, moving_rate=0.9):
        """
        Add the logic of calculating and setting output scales of some layers. 

        Args:
            moving_rate(float): The decay coefficient of moving average.
                                The default value is 0.9.
        """
        super(ImperativeCalcOutputScale, self).__init__()
        self._moving_rate = moving_rate
        self._out_scale_layer_type_list = (
            BatchNorm, BatchNorm1D, BatchNorm2D, BatchNorm3D, Conv2D, LeakyReLU,
            Linear, PReLU, Pool2D, MaxPool1D, MaxPool2D, ReLU, ReLU6, Sigmoid,
            Softmax, SyncBatchNorm, Tanh, Swish)
        self._register_hook_handle_list = []
        self._out_scale_dict = collections.OrderedDict()

    def apply(self, model):
        """
        Insert the `moving_average_abs_max_scale` op to calculate output 
        scale of specific layers in model.

        Args:
            model(fluid.dygraph.Layer): The target model which would be
            calculate the output quantization scale.

        Returns:
            None
        """
        assert isinstance(model, dygraph.Layer), \
            "The model must be the instance of dygraph.Layer."
        for _, layer in model.named_sublayers():
            if self._is_target_layer(layer):
                self._add_new_parameters(layer)
                forward_post_hook_handle = layer.register_forward_post_hook(
                    self._forward_post_hook)
                self._register_hook_handle_list.append(forward_post_hook_handle)

    def save_quantized_model(self, layer, path, input_spec=None, **config):
        """
        Save the quantized model for the inference.

        Args:
            layer (Layer): The Layer to be saved.
            path (str): The path prefix to save model. The format is ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input of the saved model's forward 
                method, which can be described by InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of the saved model. Default None.
            **configs (dict, optional): Other save configuration options for compatibility. We do not 
                recommend using these configurations, they may be removed in the future. If not necessary, 
                DO NOT use them. Default None.
                The following options are currently supported:
                (1) output_spec (list[Tensor]): Selects the output targets of the saved model.
                By default, all return variables of original Layer's forward method are kept as the 
                output of the saved model. If the provided ``output_spec`` list is not all output variables, 
                the saved model will be pruned according to the given ``output_spec`` list. 

        Returns:
            None
        """

        assert isinstance(
            layer, dygraph.Layer), "model must be the instance of dygraph.Layer"
        self._layer = layer
        is_dynamic_mode = False
        with dygraph.guard():
            self._layer.eval()
            if self._register_hook_handle_list is not None:
                for handle in self._register_hook_handle_list:
                    handle.remove()
            if self._out_scale_dict:
                for key in self._out_scale_dict:
                    self._out_scale_dict[key] = float(self._out_scale_dict[key]
                                                      .numpy())
            else:
                for _, sub_layer in self._layer.named_sublayers():
                    if self._is_target_layer(sub_layer):
                        layer_name = sub_layer.full_name()
                        if hasattr(sub_layer, "layer_name"):
                            layer_name = sub_layer.layer_name
                        if hasattr(sub_layer, "_quant_out_scale"):
                            self._out_scale_dict[layer_name] = float(
                                sub_layer._quant_out_scale)

        if paddle.in_dynamic_mode():
            is_dynamic_mode = True
            paddle.enable_static()

        paddle.jit.save(layer=layer, path=path, input_spec=input_spec, **config)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)

        file_prefix = os.path.basename(path)
        dirname = os.path.dirname(path)
        model_filename = file_prefix + INFER_MODEL_SUFFIX
        params_filename = file_prefix + INFER_PARAMS_SUFFIX

        [inference_program, feed_target_names, fetch_targets] = (
            load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        check_behind_op = False
        op_count = 0
        ops_list = [key for key, _ in self._out_scale_dict.items()]
        if len(ops_list) == 0:
            warnings.warn(
                "Warning: No Layer of the model while to be saved contains the out_threshold attribute, "
                "so the generated inference model would not contain the out_threshold."
            )
        else:
            # Because the Layer in dygraph may correspond to multiple ops
            # in static program after being saved. To ensure correctness,
            # the outscale collected for output of dygraph Layer can only
            # be set to the last op in the corresponding ops in static program.
            #
            # We can judge the execution order of the ops which corresponding
            # to dygraph Layer by check_behind_op
            forward_op = None
            for block in inference_program.blocks:
                for op in block.ops:
                    if op.type in utils._op_real_in_out_name:
                        if op_count > len(ops_list):
                            warnings.warn(
                                "The number of Layer which has out_threshold attribute should be bigger than the op in inference model"
                            )
                            break
                        if check_behind_op:
                            check_behind_op = False
                            if op.type == "elementwise_add":
                                if self._is_op_matched(ops_list[op_count], op,
                                                       block):
                                    op._set_attr("out_threshold",
                                                 self._out_scale_dict[ops_list[
                                                     op_count]])
                                    op_count += 1
                                    forward_op = None
                                continue
                            else:
                                if forward_op is None:
                                    raise ValueError(
                                        "forward_op should not be None")
                                if self._is_op_matched(ops_list[op_count],
                                                       forward_op, block):
                                    forward_op._set_attr(
                                        "out_threshold", self._out_scale_dict[
                                            ops_list[op_count]])
                                    op_count += 1
                                    forward_op = None

                        if op.type in ["conv2d", "depthwise_conv2d", "matmul"]:
                            check_behind_op = True
                            forward_op = op
                            continue
                        if op_count >= len(ops_list):
                            warnings.warn(
                                "The number of Layer which has out_threshold attribute should be bigger than the op in inference model"
                            )
                            break
                        if self._is_op_matched(ops_list[op_count], op, block):
                            op._set_attr(
                                "out_threshold",
                                self._out_scale_dict[ops_list[op_count]])
                            op_count += 1

        # Save the processed program.
        save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=inference_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename)

        if is_dynamic_mode:
            paddle.disable_static()

    def _is_target_layer(self, layer):
        return isinstance(layer, self._out_scale_layer_type_list) \
            or 'quantized_' in layer.full_name()

    # When inferenc model is saved, the logic in hook would not be executed
    # in program translation, so that some parameters can not created in
    # __init__, which would cause the model to fail to save. Therefore, the
    # parameters creation in the hook is advanced to be exected outside the hook.
    def _add_new_parameters(self, layer, name=None):
        dtype = layer._dtype if layer._dtype is not None else "float32"
        if dtype not in ["float32", "float64"]:
            return
        scale_prefix = '{}.scale'.format(name) if name else 'outscale.scale'
        scale_name = unique_name.generate(scale_prefix)
        scale_attr = ParamAttr(
            name=scale_name, initializer=Constant(1), trainable=False)
        layer._quant_out_scale = layer.create_parameter(
            shape=[1], attr=scale_attr, dtype=dtype)
        layer._quant_out_scale.stop_gradient = True

        state_prefix = "{}.state".format(name) if name else 'outscale.state'
        state_attr = ParamAttr(
            name=unique_name.generate(state_prefix),
            initializer=Constant(1),
            trainable=False)
        layer._quant_out_state = layer.create_parameter(
            shape=[1], attr=state_attr, dtype=dtype)
        layer._quant_out_state.stop_gradient = True

        accum_prefix = "{}.accum".format(name) if name else 'outscale.accum'
        accum_attr = ParamAttr(
            name=unique_name.generate(accum_prefix),
            initializer=Constant(1),
            trainable=False)
        layer._quant_out_accum = layer.create_parameter(
            shape=[1], attr=accum_attr, dtype=dtype)
        layer._quant_out_accum.stop_gradient = True

    # Judge whether the op in program matches the Layer in dynamic model
    def _is_op_matched(self, layer_name, op, block):
        output_var_names = quantization_pass._get_op_output_var_names(op)
        for output_var_name in output_var_names:
            output_var_tensor = block.var(output_var_name)
            if output_var_tensor.dtype not in [
                    core.VarDesc.VarType.FP64, core.VarDesc.VarType.FP32
            ]:
                return False

        # Because the naming styles of static and dynamic graph are different,
        # in order to avoid mistakes, we unify the name here.
        op_type = output_var_names[0].split(".")[0]
        op_type = op_type.rsplit("_", 1)[0]
        if op_type == 'depthwise_conv2d':
            op_type = 'conv2d'
        if 'prelu' in op_type:
            op_type = op_type.replace('prelu', 'p_re_lu')
        if 'relu' in op_type:
            op_type = op_type.replace('relu', 're_lu')
        return op_type in layer_name

    def _forward_post_hook(self, layer, input, output):
        assert isinstance(
            output, (core.VarBase, framework.Variable)
        ), "Multiple outputs are not currently supported in ImperativeOutScale."
        if output.dtype not in [
                core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
        ]:
            return
        if not hasattr(layer, "_out_scale"):
            self._out_scale = quant_nn.MovingAverageAbsMaxScale(
                layer, output.name, self._moving_rate, output.dtype)
        scale_out = self._out_scale(output)
        if hasattr(layer, 'layer_name'):
            layer_name = layer.layer_name
        else:
            layer_name = layer.full_name()
        self._out_scale_dict[layer_name] = scale_out
