#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import os
import numpy as np

import paddle
import paddle.nn.quant.quant_layers as quant_layers
from paddle.fluid.log_helper import get_logger
from paddle.fluid.dygraph.io import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

from . import utils
from . import ptq_hooks
from . import ptq_config
from .ptq_registry import PTQRegistry

__all__ = ['ImperativePTQ']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativePTQ(object):
    """
    Static post training quantization.
    """

    def __init__(self, quant_config=ptq_config.default_ptq_config):
        """
        Constructor.

        Args:
            quant_config(PTQConfig): the config of post training quantization.
                The config has weight_quantizer and activation_quantizer.
                In default, the weight_quantizer and activation_quantizer are
                AbsmaxQuantizer.
        """
        super(ImperativePTQ, self).__init__()

        assert isinstance(quant_config, ptq_config.PTQConfig)

        self._quant_config = quant_config

    def quantize(self, model, inplace=False):
        """
        Add hook to the leaf layer to calculate the threshold of inputs and outputs.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
            inplace(bool): Whether apply quantization to the input model.
                           Default: False.
        Returns:
            quantized_model(paddle.nn.Layer): The quantized model.
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        if not inplace:
            new_model = copy.deepcopy(model)

        for name, layer in new_model.named_sublayers():
            if PTQRegistry.is_supported_layer(layer) \
                and utils.is_leaf_layer(layer) \
                and not self._is_skip_layer(layer):
                # Add quant config
                quant_config = copy.deepcopy(self._quant_config)
                layer._quant_config = quant_config

                # register hook
                hook = ptq_hooks.quant_forward_post_hook
                quant_hook_handle = layer.register_forward_post_hook(hook)
                quant_config.quant_hook_handle = quant_hook_handle
                layer._forward_post_hooks.move_to_end(
                    quant_hook_handle._hook_id, last=False)

                # TODO(jc): fake quantize the weights

        return new_model

    def save_quantized_model(self, model, path, input_spec=None, **config):
        """
        Save the quantized model for the inference.

        Args:
            model (Layer): The model to be saved.
            path (str): The path prefix to save model. The format is 
                ``dirname/file_prefix`` or ``file_prefix``.
            input_spec (list[InputSpec|Tensor], optional): Describes the input
                of the saved model's forward method, which can be described by
                InputSpec or example Tensor. If None, all input variables of 
                the original Layer's forward method would be the inputs of
                the saved model. Default None.
            **configs (dict, optional): Other save configuration options for
                compatibility. We do not recommend using these configurations,
                they may be removed in the future. If not necessary, DO NOT use
                them. Default None.
                The following options are currently supported:
                (1) output_spec (list[Tensor]): Selects the output targets of
                the saved model. By default, all return variables of original
                Layer's forward method are kept as the output of the saved model.
                If the provided ``output_spec`` list is not all output variables, 
                the saved model will be pruned according to the given
                ``output_spec`` list. 

        Returns:
            None
        """

        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        model = self._post_process_scales(model)
        model = self._wrap_layers(model)

        paddle.jit.save(layer=model, path=path, input_spec=input_spec, **config)

        is_dynamic_mode = False
        if paddle.in_dynamic_mode():
            is_dynamic_mode = True
            paddle.enable_static()

        place = paddle.CPUPlace()
        scope = paddle.static.global_scope()
        exe = paddle.static.Executor(place)

        dirname = os.path.dirname(path)
        basename = os.path.basename(path)
        model_filename = basename + INFER_MODEL_SUFFIX
        params_filename = basename + INFER_PARAMS_SUFFIX

        [infer_program, feed_target_names, fetch_targets] = (
            paddle.fluid.io.load_inference_model(
                dirname=dirname,
                executor=exe,
                model_filename=model_filename,
                params_filename=params_filename))

        # TODO(jc): 
        # process the first moving_average_abs_max_scale layer
        # fuse conv + bn
        # propagate the threshold  
        # support skip_quant

        # self._save_output_scale(infer_program, scope)
        # self._set_skip_quant_attr(infer_program)

        paddle.fluid.io.save_inference_model(
            dirname=dirname,
            feeded_var_names=feed_target_names,
            target_vars=fetch_targets,
            executor=exe,
            main_program=infer_program.clone(),
            model_filename=model_filename,
            params_filename=params_filename)

        if is_dynamic_mode:
            paddle.disable_static()

    def _post_process_scales(self, model):
        """
        Process the scales and remove the hooks.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            converted_model(paddle.nn.Layer): The converted model.
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        # remove hook and calculate thresholds
        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                quant_config = sub_layer._quant_config
                quant_config.quant_hook_handle.remove()

                quant_config.out_act_quantizer.cal_thresholds()

                if PTQRegistry.is_simulated_quant_layer(sub_layer):
                    weights = (sub_layer.weight, )
                    quant_config.wt_quantizer.sample_data(sub_layer, weights)
                    quant_config.wt_quantizer.cal_thresholds()

        # save output activation and weight thresholds
        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer):
                quant_config = sub_layer._quant_config
                layer_info = PTQRegistry.layer_info(sub_layer)

                output_names = layer_info.output_names
                output_thresholds = quant_config.out_act_quantizer.thresholds
                assert len(output_names) == 1
                assert len(output_thresholds) == 1
                save_name = output_names[0] + str(0) + "_threshold"
                sub_layer._set_op_attrs({save_name: output_thresholds[0]})
                sub_layer._set_op_attrs({"out_threshold": output_thresholds[0]})

                if PTQRegistry.is_simulated_quant_layer(sub_layer):
                    weight_names = layer_info.weight_names
                    weight_thresholds = quant_config.wt_quantizer.thresholds
                    assert len(weight_names) == 1
                    assert len(weight_thresholds) == 1
                    save_name = weight_names[0] + str(0) + "_threshold"
                    sub_layer._set_op_attrs({save_name: weight_thresholds[0]})

        return model

    def _wrap_layers(self, model):
        """
        Replace conv2d and linear with the quantized layers, and save
        thresholds into the fake layers.
        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            modified_model(paddle.nn.Layer): The modified model.
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        # wrap conv2d and linear, save thresholds and quant bits to fake ops
        for name, sub_layer in model.named_sublayers():
            if self._is_quant_layer(sub_layer) \
                and PTQRegistry.is_simulated_quant_layer(sub_layer):
                parent_layer, sub_name = \
                    utils.find_parent_layer_and_sub_name(model, name)

                quant_layer_name = None
                for key, value in utils.layer_name_map.items():
                    if isinstance(sub_layer, value):
                        quant_layer_name = 'Quantized' + key
                        break
                assert quant_layer_name is not None

                # TODO(jc):
                # quant_layer = quant_layers.__dict__[quant_layer_name](sub_layer, **self._kwargs)
                # setattr(parent_layer, sub_name, quant_layer)

        return model

    @staticmethod
    def _is_skip_layer(layer):
        return hasattr(layer, "skip_quant") and layer.skip_quant == True

    @staticmethod
    def _is_quant_layer(layer):
        return hasattr(layer, "_quant_config")
