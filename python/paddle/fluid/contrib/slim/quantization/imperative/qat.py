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

import logging
import numpy as np
import sys
from paddle.fluid import dygraph, core
from paddle.fluid.dygraph.nn import Conv2D, Linear, BatchNorm, Pool2D, Conv2DTranspose, PRelu
from paddle.nn.layer import ReLU, LeakyReLU, Sigmoid, ReLU6, Tanh, Softmax, PReLU
from paddle.fluid.log_helper import get_logger
from . import quant_nn

__all__ = ['ImperativeQuantAware', 'ImperativeOutScale']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativeQuantAware(object):
    """
    Add the fake quant logic for given quantizable layers, namely add the quant_dequant
    computational logic both for activation inputs and weight inputs.
    """

    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='moving_average_abs_max',
                 moving_rate=0.9,
                 quantizable_layer_type=['Conv2D', 'Linear']):
        """
        The constructor for ImperativeQuantAware.

        Args:
            weight_bits(int): quantization bit number for weights,
                whereas the bias is not quantized.
            activation_bits(int): quantization bit number for activations.
            weight_quantize_type(str): quantization type for weights,
                which supports 'abs_max' now. The 'moving_average_abs_max'
                usually is not used for weights, since weights are fixed once the
                model is well trained.
            activation_quantize_type(str): quantization type for activations,
                which supports 'abs_max' and 'moving_average_abs_max' now.
                If using 'abs_max' mode, the quantization scale will be calculated
                dynamically each step in both training and testing period. If using
                'moving_average_abs_max', the static quantization scale will be calculated
                during training and used in inference.
            moving_rate(float): the parameter for 'moving_average_abs_max' quantization.
            quantizable_op_type(list[str]): List the type of layers that will be quantized. 
                Default is ['Conv2D', 'Linear']. The quantizable_op_type in
                QuantizationFreezePass and ConvertToInt8Pass must be the same as this.


        Examples:
        .. code-block:: python

            from paddle.fluid.contrib.slim.quantization \
                import ImperativeQuantAware
            from paddle.incubate.hapi.vision.models \
                import resnet
            
            model = resnet.resnet50(pretrained=True)

            imperative_qat = ImperativeQuantAware(
                weight_quantize_type='abs_max',
                activation_quantize_type='moving_average_abs_max')
            
            # Add the fake quant logical.
            # The original model will be rewrite.
            imperative_qat.quantize(model)

            # Fine-tune the quantized model
            # ...
            
            # Save quant model for the inference.
            imperative_qat.save_quantized_model(
                dirname="./resnet50_qat",
                model=model,
                input_shape=[(3, 224, 224)],
                input_dtype=['float32'],
                feed=[0],
                fetch=[0])
        """
        super(ImperativeQuantAware, self).__init__()
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._moving_rate = moving_rate

        quant_type = {'abs_max', 'moving_average_abs_max'}
        if activation_quantize_type not in quant_type:
            raise ValueError(
                "Unknown activation_quantize_type : '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' now." %
                (str(activation_quantize_type)))
        if weight_quantize_type not in quant_type:
            raise ValueError(
                "Unknown weight_quantize_type: '%s'. It can only be "
                "'abs_max' or 'moving_average_abs_max' now." %
                (str(weight_quantize_type)))
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type

        self._quant_layers_map = {'Conv2D': Conv2D, 'Linear': Linear}
        self._quantizable_layer_type = tuple(
            self._quant_layers_map[layer]
            if layer in self._quant_layers_map else layer
            for layer in quantizable_layer_type)
        for layer in self._quantizable_layer_type:
            assert not isinstance(
                layer, str), "{} is unspported to be quantized.".format(layer)

    def quantize(self, model):
        """
        According to weights' and activations' quantization types, the model will be added some fake
        quant ops, such as fake_quantize_dequantize_moving_average_abs_max, fake_quantize_dequantize_abs_max
        and so on.

        Args:
            model(fluid.dygraph.Layer): the model to be quantized.
        Returns:
            None
        """
        for name, layer in model.named_sublayers():
            if not isinstance(layer, self._quantizable_layer_type):
                continue
            scopes = name.split('.')
            target = scopes[-1]
            obj = model
            parent = model
            for i in range(len(scopes) - 1):
                obj = getattr(parent, scopes[i])
                parent = obj

            quant_layer = self._get_quantized_counterpart(layer)
            setattr(obj, target, quant_layer)

    def save_quantized_model(self,
                             dirname,
                             model,
                             input_shape,
                             input_dtype,
                             feed,
                             fetch,
                             append_batch_size=True):
        """
        Save the quantized model for the inference.

        Args:
            dirname (str): the directory to save the quantized model.
            model(fluid.dygraph.Layer): the quantized model to be saved.
            input_shape(list[tuple(int)]): The shape value for each input,
                e.g. [(3, 224, 224)].
            input_dtype(list[str]): The dtype value for each input,
                e.g. ['float32'].
            feed(list[int]): the indices of the input variables of the
                imperative functions which will be saved as input variables in
                inference model.
            fetch(list[int]): the indices of the returned variable of the
                imperative functions which will be saved as output variables in
                inference model.
            append_batch_size(bool, optional):
                If true, it prepends an extra axis to the input_shape, meanwhile,
                the input_shape shouldn't contain the batch size dimension.
                Otherwise, it just uses the input_shape. Default True.
        Returns:
            None
        """
        assert isinstance(
            input_shape, list), "The parameter `input_shape` shoubld be a list."
        assert isinstance(
            input_dtype, list), "The parameter `input_dtype` shoubld be a list."
        assert isinstance(feed, list), "The parameter `feed` shoubld be a list."
        assert isinstance(fetch,
                          list), "The parameter `fetch` shoubld be a list."
        assert len(input_shape) == len(
            input_dtype
        ), "The length of input_shape should be equal to  input_dtype's."
        assert len(input_dtype) == len(
            feed), "The length of input_shape should be equal to  feed's."

        prog_trans = dygraph.ProgramTranslator()
        with dygraph.guard():
            model.eval()
            input_vars = []
            for shape, dtype in zip(input_shape, input_dtype):
                raw_data = np.random.random(shape)
                input_data = raw_data[np.newaxis, :].astype(
                    dtype) if append_batch_size else raw_data.astype(dtype)
                input_var = dygraph.to_variable(input_data)
                input_vars.append(input_var)
            outputs = prog_trans.get_output(model.forward, model, *input_vars)
        input_spec = [input_vars[i] for i in feed]
        configs = dygraph.jit.SaveLoadConfig()
        configs.separate_params = True
        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]
        configs.output_spec = [outputs[i] for i in fetch]
        dygraph.jit.save(
            layer=model,
            model_path=dirname,
            input_spec=input_spec,
            configs=configs)

    def _get_quantized_counterpart(self, layer):
        quant_layers = tuple(self._quant_layers_map.values())
        quantized_counterpart = tuple('Quantized' + k
                                      for k in self._quant_layers_map.keys())

        predicate = lambda value: isinstance(layer, value)
        index_generator = (i for i, v in enumerate(quant_layers)
                           if predicate(v))

        try:
            index = next(index_generator)
        except StopIteration:
            _logger.fatal("The layer {} is unsupported to be quantized.".format(
                layer.full_name()))
            sys.exit(-1)

        quantized_layer = quant_nn.__dict__[quantized_counterpart[index]](
            layer, self._weight_bits, self._activation_bits, self._moving_rate,
            self._weight_quantize_type, self._activation_quantize_type)
        return quantized_layer


class ImperativeOutScale(object):
    def __init__(self,
                 moving_rate=0.9,
                 out_scale_layer_type=[
                     'Conv2D', 'Pool2D', 'ReLU', 'BatchNorm', 'PRelu',
                     'Sigmoid', 'LeakyReLU', 'ReLU6', 'Tanh', 'Softmax',
                     'Conv2DTranspose'
                 ]):
        """
        Add the logic of calculating and setting output quantization scales of some layers.
        These output quantization scales may be used by tensorRT or some other inference engines.

        Args:
            moving_rate(float): the parameter for 'moving_average_abs_max_scale' quantization.
            quantizable_op_type(list[str]): List the type of layers that will be got out_scale. 
                Default is ['Conv2D', 'ReLU', 'PReLU', 'LeakyReLU', 'Sigmoid', 'BatchNorm', 'ReLU6', 'Tanh', 'Softmax', 'Conv2DTranspose']
        """
        super(ImperativeOutScale, self).__init__()
        self._moving_rate = moving_rate
        self._out_scale_layers_map = {
            'Conv2D': Conv2D,
            'Pool2D': Pool2D,
            'ReLU': ReLU,
            'LeakyReLU': LeakyReLU,
            'Sigmoid': Sigmoid,
            'PRelu': PRelu,
            'BatchNorm': BatchNorm,
            'ReLU6': ReLU6,
            'Tanh': Tanh,
            'Softmax': Softmax,
            'Conv2DTranspose': Conv2DTranspose
        }
        self._out_scale_layer_type = tuple(
            self._out_scale_layers_map[layer]
            if layer in self._out_scale_layers_map else layer
            for layer in out_scale_layer_type)
        for layer in self._out_scale_layer_type:
            assert not isinstance(
                layer, str), "{} is unspported to be out_scaled.".format(layer)

    def get_out_scale(self, model):
        """
        Insert the `moving_average_abs_max_scale` op to calculate output scale of Specific layers in model.

        Args:
            model(fluid.dygraph.Layer): The target model which would be calculate the output quantization scale.

        Returns:
            None
        """
        self._register_hook_handle_list = []
        assert isinstance(
            model, dygraph.Layer), "model must be the instance of dygraph.Layer"
        self._out_scale_dict = {}
        for _, layer in model.named_sublayers():
            if not isinstance(layer, self._out_scale_layer_type):
                continue
            forward_post_hook_handle = layer.register_forward_post_hook(
                self._forward_post_hook)
            self._register_hook_handle_list.append(forward_post_hook_handle)

    def set_out_scale(self, model):
        """
        Set the output quantization scale to the corresponding Layer of model.

        Args:
            model(fluid.dygraph.Layer): The output scale would be added to the model.

        Returns:
            None
        """
        assert isinstance(
            model, dygraph.Layer), "model must be the instance of dygraph.Layer"
        for _, layer in model.named_sublayers():
            if not isinstance(layer, self._out_scale_layer_type):
                continue
            scale_out_list = self._out_scale_dict[layer.full_name()]
            for scale_var in scale_out_list:
                layer.__setattr__('out_threshold', scale_var)
        for handle in self._register_hook_handle_list:
            handle.remove()

    def _forward_post_hook(self, layer, input, output):
        if not isinstance(output, list):
            output = [output]
        scale_out_list = []
        for var in output:
            if var.dtype not in [
                    core.VarDesc.VarType.FP32, core.VarDesc.VarType.FP64
            ]:
                continue
            if not hasattr(layer, "_out_scale"):
                layer._out_scale = quant_nn.MovingAverageAbsMaxScale(
                    var.name, self._moving_rate, var.dtype)
            scale_out = layer._out_scale(var)
            scale_out_list.append(float(scale_out.numpy()))
        self._out_scale_dict[layer.full_name()] = scale_out_list
