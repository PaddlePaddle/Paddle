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
import numpy as np

import paddle
from paddle.fluid.log_helper import get_logger

from . import utils
from . import ptq_hooks

__all__ = ['ImperativePTQ']

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class QuantConfig:
    """
    The quant config saved in layers.
    """

    def __init__(self, forward_post_hook=None):
        self.forward_post_hook = forward_post_hook
        self.input_thresholds = []
        self.output_thresholds = []
        # hist
        self.input_abs_max_vals = []
        self.output_abs_max_vals = []
        self.input_hists = []
        self.output_hists = []
        self.bins = 1024
        self.upsample_bins = 64
        self.hist_percent = 0.99999


class ImperativePTQ(object):
    """
    Applying static post_training quantization to dgraph model.
    """

    def __init__(self, algo="abs_max", activation_bits=8, weight_bits=8):
        """
        Constructor.
        Args:
            algo(str): The algorithm in post_training quantizaion to be used.
            activation_bits(int): quantization bit number for activations.
            weight_bits(int): quantization bit number for weights.
        """
        super(ImperativePTQ, self).__init__()

        supported_algo = ["abs_max", "hist", "kl"]
        assert algo in supported_algo, \
            "Input algo error: algo should be one of " + str(supported_algo)

        self._algo = algo
        self._activation_bits = activation_bits
        self._weight_bits = weight_bits

    def quantize(self, model):
        """
        Add hook to the leaf layer to calculate the threshold of inputs and outputs.

        Args:
            model(paddle.nn.Layer): The model to be quantized.
        Returns:
            None
        """
        assert isinstance(model, paddle.nn.Layer), \
            "The model must be the instance of paddle.nn.Layer."

        for name, layer in model.named_sublayers():
            if utils.is_leaf_layer(layer):
                hook_name = self._algo + "_forward_post_hook"
                assert hook_name in ptq_hooks.__dict__
                hook_handle = layer.register_forward_post_hook(
                    ptq_hooks.__dict__[hook_name])
                layer._forward_post_hooks.move_to_end(
                    hook_handle._hook_id, last=False)
                layer._quant_config = QuantConfig(forward_post_hook=hook_handle)

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

                if self._algo == "hist":
                    self._post_process_hist(sub_layer)
                elif self._algo == "kl":
                    self._post_process_kl(sub_layer)

                # TODO (jc): 
                # get the threshold of weight for conv2d and linear
                # add fake_quant_ops
                # save thresholds
                quant_config = sub_layer._quant_config
                quant_config.forward_post_hook.remove()

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
                    self._activation_bits)
                qc.input_thresholds.append(threshold)

        for idx in range(len(qc.output_hists)):
            if qc.output_hists[idx] is None:
                qc.output_thresholds.append(qc.output_abs_max_vals[idx])
            else:
                threshold = utils.cal_kl_scaling_factor(
                    qc.output_hists[idx], qc.output_abs_max_vals[idx],
                    self._activation_bits)
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
