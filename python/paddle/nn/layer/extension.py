# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: define the common classes to build a neural network
import paddle
from ...fluid.dygraph import Flatten  # noqa: F401
from ...fluid.framework import in_dygraph_mode
from .. import functional as F
from ...fluid.framework import _dygraph_tracer
from paddle.nn import Layer

__all__ = []


class BezierAlign(Layer):
    r"""
    Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling bezier_align. This produces the correct neighbors; see
            adet/tests/test_bezier_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
    """

    # TODO: update doc
    def __init__(self,
                 output_size,
                 spatial_scale,
                 sampling_ratio,
                 aligned=True,
                 **kwargs):
        super(BezierAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois, rois_num):
        return F.bezier_align(
            input,
            rois,
            rois_num,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio,
            aligned=self.aligned)
