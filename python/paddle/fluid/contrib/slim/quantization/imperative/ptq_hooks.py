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

import paddle
import math
import numpy as np


def abs_max_value(tensor):
    return float(paddle.max(paddle.abs(tensor)).numpy())


def merge_max_value(old, new):
    if old != []:
        assert len(old) == len(new)
        for i in range(len(old)):
            new[i] = old[i] if new[i] < old[i] else new[i]
    return new


def abs_max_forward_post_hook(layer, inputs, outputs):
    """
    The forward post hook for calculating abs max value.
    """

    assert hasattr(layer, '_quant_config'), \
        "The layer should have _quant_config attr"

    # The input num maybe greater than 1, the output num is 1
    in_threshold = [abs_max_value(t) for t in inputs]
    out_threshold = [max([abs_max_value(t) for t in outputs])]

    quant_config = layer._quant_config
    quant_config.input_thresholds = \
        merge_max_value(quant_config.input_thresholds, in_threshold)
    quant_config.output_thresholds = \
        merge_max_value(quant_config.output_thresholds, out_threshold)

    return outputs


def kl_forward_post_hook(layer, inputs, outputs):
    """
    The forward post hook for using KL to calculate threshods.
    """

    def _combine_max_and_hist(tensor, origin_max, origin_hist, bins,
                              upsample_bins):
        """
        """

        new_max = abs_max_value(tensor)

        if new_max == 0.0:
            return origin_max, origin_hist
        elif origin_max == 0.0:
            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(), range=(0, new_max), bins=bins)
            new_hist = new_hist.astype(np.float32)
            return new_max, new_hist
        elif new_max <= origin_max:
            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(), range=(0, origin_max), bins=bins)
            new_hist = new_hist.astype(np.float32)
            new_hist += origin_hist
            return origin_max, new_hist
        else:
            # bin_width = origin_max / (bins * upsample_bins) 
            #           = new_max / (bins * downsample_bins)
            bin_width = origin_max / (bins * upsample_bins)
            downsampe_bins = int(math.ceil(new_max / (bins * bin_width)))
            new_max = bins * bin_width * downsampe_bins

            upsampled_hist = np.repeat(origin_hist, upsample_bins)
            expanded_hist = np.zeros((bins * downsampe_bins), dtype=np.float32)
            expanded_hist[0:bins * upsample_bins] = upsampled_hist
            cumsumed_hist = np.cumsum(
                expanded_hist,
                dtype=np.float64)[downsampe_bins - 1::downsampe_bins]
            shift_cumsumed_hist = np.zeros((bins), dtype=np.float64)
            shift_cumsumed_hist[1:] = cumsumed_hist[0:-1]
            sampled_hist = (cumsumed_hist - shift_cumsumed_hist) / upsample_bins
            sampled_hist = sampled_hist.astype(np.float32)

            new_hist, _ = np.histogram(
                paddle.abs(tensor).numpy(), range=(0, new_max), bins=bins)
            new_hist = new_hist.astype(np.float32)
            new_hist += sampled_hist

            return new_max, new_hist

    assert hasattr(layer, '_quant_config'), \
        "The layer should have _quant_config attr"
    quant_config = layer._quant_config
    bins = quant_config.bins
    upsample_bins = quant_config.upsample_bins

    if quant_config.input_abs_max_vals == [] \
        or quant_config.output_abs_max_vals == []:
        in_max = [abs_max_value(t) for t in inputs]
        out_max = [max([abs_max_value(t) for t in outputs])]

        quant_config.input_abs_max_vals = in_max
        quant_config.output_abs_max_vals = out_max

        for idx, in_tensor in enumerate(inputs):
            if in_max[idx] == 0.0:
                quant_config.input_hists.append(None)
            else:
                hist, _ = np.histogram(
                    paddle.abs(in_tensor).numpy(),
                    range=(0., in_max[idx]),
                    bins=bins)
                hist = hist.astype(np.float32)
                quant_config.input_hists.append(hist)

        if out_max[0] == 0.0:
            quant_config.output_hists.append(None)
        else:
            hist, _ = np.histogram(
                paddle.abs(outputs).numpy(), range=(0., out_max[0]), bins=bins)
            hist = hist.astype(np.float32)
            quant_config.output_hists.append(hist)
    else:
        for idx, tensor in enumerate(inputs):
            new_max, new_hist = _combine_max_and_hist(
                tensor, quant_config.input_abs_max_vals[idx],
                quant_config.input_hists[idx], bins, upsample_bins)
            quant_config.input_abs_max_vals[idx] = new_max
            quant_config.input_hists[idx] = new_hist

        new_max, new_hist = _combine_max_and_hist(
            outputs, quant_config.output_abs_max_vals[0],
            quant_config.output_hists[0], bins, upsample_bins)
        quant_config.output_abs_max_vals[0] = new_max
        quant_config.output_hists[0] = new_hist

    return outputs


hist_forward_post_hook = kl_forward_post_hook
