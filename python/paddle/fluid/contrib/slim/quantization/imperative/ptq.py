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
import numpy as np

import paddle
from paddle.fluid.log_helper import get_logger

from . import utils
from . import ptq_hooks
from . import ptq_config

__all__ = ['ImperativePTQ']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class ImperativePTQ(object):
    """
    Applying static post_training quantization to dgraph model.
    """

    def __init__(self, quant_config=ptq_config.AbsMaxPTQConfig()):
        """
        Constructor.
        Args:
            algo(str): The algorithm in post_training quantizaion to be used.
            activation_bits(int): quantization bit number for activations.
            weight_bits(int): quantization bit number for weights.
        """
        super(ImperativePTQ, self).__init__()

        assert isinstance(quant_config, ptq_config.PTQConfig)

        self._quant_config = quant_config

    def quantize(self, model, inplace=False):
        """
        Add hook to the leaf layer to calculate the threshold of inputs and outputs.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        if not inplace:
            model = copy.deepcopy(model)

        for name, layer in model.named_sublayers():
            if utils.is_leaf_layer(layer):
                quant_config = copy.deepcopy(self._quant_config)
                hook = layer.register_forward_post_hook(quant_config.get_hook())
                layer._forward_post_hooks.move_to_end(hook._hook_id, last=False)
                quant_config.set_hook(hook)
                layer._quant_config = quant_config

        return model

    def convert(self, model):
        """
        Process the scales and remove the hooks.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The input model must be the instance of paddle.nn.Layer."

        for name, sub_layer in model.named_sublayers():
            if utils.is_leaf_layer(sub_layer):
                assert hasattr(sub_layer, "_quant_config")

                if isinstance(self._quant_config, ptq_config.HistPTQConfig):
                    self._post_process_hist(sub_layer)
                elif isinstance(self._quant_config, ptq_config.KLPTQConfig):
                    self._post_process_kl(sub_layer)

                # TODO (jc): 
                # get the threshold of weight for conv2d and linear
                # add fake_quant_ops
                # save thresholds
                quant_config = sub_layer._quant_config
                quant_config.hook.remove()

    def _post_process_kl(self, layer):
        """
        Post process for using kl algorithm.
        """
        _logger.info("Post process for %s" % layer.full_name())

        qc = layer._quant_config
        for idx in range(len(qc.input_hists)):
            if qc.input_hists[idx] is None:
                qc.input_thresholds.append(qc.input_abs_max_vals[idx])
            else:
                threshold = utils.cal_kl_scaling_factor(
                    qc.input_hists[idx], qc.input_abs_max_vals[idx],
                    qc.activation_bits)
                qc.input_thresholds.append(threshold)

        for idx in range(len(qc.output_hists)):
            if qc.output_hists[idx] is None:
                qc.output_thresholds.append(qc.output_abs_max_vals[idx])
            else:
                threshold = utils.cal_kl_scaling_factor(
                    qc.output_hists[idx], qc.output_abs_max_vals[idx],
                    qc.activation_bits)
                qc.output_thresholds.append(threshold)

    def _post_process_hist(self, layer):
        """
        Post process for using kl algorithm.
        """

        def _helper(abs_max, hist, percent):
            assert hist.ndim == 1 and percent < 1.0
            hist = hist / np.sum(hist, dtype=np.float64)
            cumsumed_hist = np.cumsum(hist)
            index = np.argwhere(cumsumed_hist >= percent)[0]
            return float((index - 0.5) * (abs_max / hist.shape[0]))

        qc = layer._quant_config
        for idx in range(len(qc.input_hists)):
            if qc.input_hists[idx] is None:
                qc.input_thresholds.append(qc.input_abs_max_vals[idx])
            else:
                threshold = _helper(qc.input_abs_max_vals[idx],
                                    qc.input_hists[idx], qc.hist_percent)
                qc.input_thresholds.append(threshold)

        for idx in range(len(qc.output_hists)):
            if qc.output_hists[idx] is None:
                qc.output_thresholds.append(qc.output_abs_max_vals[idx])
            else:
                threshold = _helper(qc.output_abs_max_vals[idx],
                                    qc.output_hists[idx], qc.hist_percent)
                qc.output_thresholds.append(threshold)
