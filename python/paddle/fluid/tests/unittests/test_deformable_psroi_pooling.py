# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import Program, program_guard


def set_input(input, rois, trans):
    inputs = {'Input': input, "ROIs": rois, "Trans": trans}
    return inputs


def set_attrs(no_trans, spatial_scale, output_channels, group_size,
              pooled_height, pooled_width, part_size, sample_per_part,
              trans_std):
    attrs = {
        'no_trans': no_trans,
        'spatial_scale': spatial_scale,
        'output_dim': output_channels,
        'group_size': group_size,
        'pooled_height': pooled_height,
        'pooled_width': pooled_width,
        'part_size': part_size,
        'sample_per_part': sample_per_part,
        'trans_std': trans_std
    }
    return attrs


def set_outputs(output, top_count):
    outputs = {
        'Output': output.astype('float32'),
        'TopCount': top_count.astype('float32')
    }
    return outputs


class TestDeformablePSROIPoolOp(OpTest):

    def set_data(self):
        self.start_test1()
        self.start_test2()
        self.start_test3()
        self.start_test4()

    def start_test1(self):
        self.init_test_case1()
        self.make_rois()
        self.calc_deformable_psroi_pooling()

        inputs = self.input
        rois = (self.rois[:, 1:5], self.rois_lod)
        trans = self.trans
        self.inputs = set_input(inputs, rois, trans)

        no_trans = self.no_trans
        spatial_scale = self.spatial_scale
        output_channels = self.output_channels
        group_size = self.group_size
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        part_size = self.part_size
        sample_per_part = self.sample_per_part
        trans_std = self.trans_std

        self.attrs = set_attrs(no_trans, spatial_scale, output_channels,
                               group_size, pooled_height, pooled_width,
                               part_size, sample_per_part, trans_std)

        output = self.out.astype('float32')
        top_count = self.top_count.astype('float32')
        self.outputs = set_outputs(output, top_count)

    def start_test2(self):
        self.init_test_case2()
        self.make_rois()
        self.calc_deformable_psroi_pooling()

        inputs = self.input
        rois = (self.rois[:, 1:5], self.rois_lod)
        trans = self.trans
        self.inputs = set_input(inputs, rois, trans)

        no_trans = self.no_trans
        spatial_scale = self.spatial_scale
        output_channels = self.output_channels
        group_size = self.group_size
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        part_size = self.part_size
        sample_per_part = self.sample_per_part
        trans_std = self.trans_std

        self.attrs = set_attrs(no_trans, spatial_scale, output_channels,
                               group_size, pooled_height, pooled_width,
                               part_size, sample_per_part, trans_std)

        output = self.out.astype('float32')
        top_count = self.top_count.astype('float32')
        self.outputs = set_outputs(output, top_count)

    def start_test3(self):
        self.init_test_case3()
        self.make_rois()
        self.calc_deformable_psroi_pooling()

        inputs = self.input
        rois = (self.rois[:, 1:5], self.rois_lod)
        trans = self.trans
        self.inputs = set_input(inputs, rois, trans)

        no_trans = self.no_trans
        spatial_scale = self.spatial_scale
        output_channels = self.output_channels
        group_size = self.group_size
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        part_size = self.part_size
        sample_per_part = self.sample_per_part
        trans_std = self.trans_std

        self.attrs = set_attrs(no_trans, spatial_scale, output_channels,
                               group_size, pooled_height, pooled_width,
                               part_size, sample_per_part, trans_std)

        output = self.out.astype('float32')
        top_count = self.top_count.astype('float32')
        self.outputs = set_outputs(output, top_count)

    def start_test4(self):
        self.init_test_case4()
        self.make_rois()
        self.calc_deformable_psroi_pooling()

        inputs = self.input
        rois = (self.rois[:, 1:5], self.rois_lod)
        trans = self.trans
        self.inputs = set_input(inputs, rois, trans)

        no_trans = self.no_trans
        spatial_scale = self.spatial_scale
        output_channels = self.output_channels
        group_size = self.group_size
        pooled_height = self.pooled_height
        pooled_width = self.pooled_width
        part_size = self.part_size
        sample_per_part = self.sample_per_part
        trans_std = self.trans_std

        self.attrs = set_attrs(no_trans, spatial_scale, output_channels,
                               group_size, pooled_height, pooled_width,
                               part_size, sample_per_part, trans_std)

        output = self.out.astype('float32')
        top_count = self.top_count.astype('float32')
        self.outputs = set_outputs(output, top_count)

    def init_test_case1(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [
            self.batch_size, self.channels, self.height, self.width
        ]
        self.no_trans = False
        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 4
        self.pooled_width = 4
        self.part_size = [4, 4]
        self.sample_per_part = 2
        self.trans_std = 0.1
        self.input = np.random.random(self.input_dim).astype('float32')

    def init_test_case2(self):
        self.batch_size = 2
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [
            self.batch_size, self.channels, self.height, self.width
        ]
        self.no_trans = True
        self.spatial_scale = 1.0 / 2.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 7
        self.pooled_width = 7
        self.part_size = [7, 7]
        self.sample_per_part = 4
        self.trans_std = 0.1
        self.input = np.random.random(self.input_dim).astype('float32')

    def init_test_case3(self):
        self.batch_size = 2
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [
            self.batch_size, self.channels, self.height, self.width
        ]
        self.no_trans = False
        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 3
        self.pooled_width = 3
        self.part_size = [3, 3]
        self.sample_per_part = 3
        self.trans_std = 0.2
        self.input = np.random.random(self.input_dim).astype('float32')

    def init_test_case4(self):
        self.batch_size = 2
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [
            self.batch_size, self.channels, self.height, self.width
        ]
        self.no_trans = True
        self.spatial_scale = 1.0 / 2.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 6
        self.pooled_width = 2
        self.part_size = [6, 6]
        self.sample_per_part = 6
        self.trans_std = 0.4
        self.input = np.random.random(self.input_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x_1 = np.random.randint(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y_1 = np.random.randint(
                    0, self.height // self.spatial_scale - self.pooled_height)
                x_2 = np.random.randint(x_1 + self.pooled_width,
                                        self.width // self.spatial_scale)
                y_2 = np.random.randint(y_1 + self.pooled_height,
                                        self.height // self.spatial_scale)
                roi = [bno, x_1, y_1, x_2, y_2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float32")

    def dmc_bilinear(self, data_im, p_h, p_w):
        h_low = int(np.floor(p_h))
        w_low = int(np.floor(p_w))
        h_high = h_low + 1
        w_high = w_low + 1
        l_h = p_h - h_low
        l_w = p_w - w_low
        h_h = 1 - l_h
        h_w = 1 - l_w
        v_1 = 0
        if h_low >= 0 and w_low >= 0:
            v_1 = data_im[h_low, w_low]
        v_2 = 0
        if h_low >= 0 and w_high <= self.width - 1:
            v_2 = data_im[h_low, w_high]
        v_3 = 0
        if h_high <= self.height - 1 and w_low >= 0:
            v_3 = data_im[h_high, w_low]
        v_4 = 0
        if h_high <= self.height - 1 and w_high <= self.width - 1:
            v_4 = data_im[h_high, w_high]
        w_1, w_2, w_3, w_4 = h_h * h_w, h_h * l_w, l_h * h_w, l_h * l_w
        val = w_1 * v_1 + w_2 * v_2 + w_3 * v_3 + w_4 * v_4
        return val

    def calc_deformable_psroi_pooling(self):
        output_shape = (self.rois_num, self.output_channels, self.pooled_height,
                        self.pooled_width)
        self.out = np.zeros(output_shape)
        self.trans = np.random.rand(self.rois_num, 2, self.part_size[0],
                                    self.part_size[1]).astype('float32')
        self.top_count = np.random.random((output_shape)).astype('float32')
        count = self.rois_num * self.output_channels * self.pooled_height * self.pooled_width
        for index in range(count):
            p_w = int(index % self.pooled_width)
            p_h = int(index / self.pooled_width % self.pooled_height)
            ctop = int(index / self.pooled_width / self.pooled_height %
                       self.output_channels)
            n_out = int(index / self.pooled_width / self.pooled_height /
                        self.output_channels)
            roi = self.rois[n_out]
            roi_batch_id = int(roi[0])
            roi_start_w = int(np.round(roi[1])) * self.spatial_scale - 0.5
            roi_start_h = int(np.round(roi[2])) * self.spatial_scale - 0.5
            roi_end_w = int(np.round(roi[3] + 1)) * self.spatial_scale - 0.5
            roi_end_h = int(np.round(roi[4] + 1)) * self.spatial_scale - 0.5
            roi_width = max(roi_end_w - roi_start_w, 0.1)
            roi_height = max(roi_end_h - roi_start_h, 0.1)
            bin_size_h = float(roi_height) / float(self.pooled_height)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            sub_bin_size_h = bin_size_h / self.sample_per_part
            sub_bin_size_w = bin_size_w / self.sample_per_part
            part_h = int(np.floor(p_h) / self.pooled_height * self.part_size[0])
            part_w = int(np.floor(p_w) / self.pooled_width * self.part_size[1])
            if self.no_trans:
                trans_x = 0
                trans_y = 0
            else:
                trans_x = self.trans[n_out][0][part_h][part_w] * self.trans_std
                trans_y = self.trans[n_out][1][part_h][part_w] * self.trans_std
            wstart = p_w * bin_size_w + roi_start_w
            wstart = wstart + trans_x * roi_width
            hstart = p_h * bin_size_h + roi_start_h
            hstart = hstart + trans_y * roi_height
            sum = 0
            num_sample = 0
            g_w = np.floor(p_w * self.group_size[0] / self.pooled_height)
            g_h = np.floor(p_h * self.group_size[1] / self.pooled_width)
            g_w = min(max(g_w, 0), self.group_size[0] - 1)
            g_h = min(max(g_h, 0), self.group_size[1] - 1)
            input_i = self.input[roi_batch_id]
            for i_w in range(self.sample_per_part):
                for i_h in range(self.sample_per_part):
                    w_sample = wstart + i_w * sub_bin_size_w
                    h_sample = hstart + i_h * sub_bin_size_h
                    if w_sample < -0.5 or w_sample > self.width - 0.5 or \
                    h_sample < -0.5 or h_sample > self.height - 0.5:
                        continue
                    w_sample = min(max(w_sample, 0.), self.width - 1.)
                    h_sample = min(max(h_sample, 0.), self.height - 1.)
                    c_sample = int((ctop * self.group_size[0] + g_h) *
                                   self.group_size[1] + g_w)
                    val = self.dmc_bilinear(input_i[c_sample], h_sample,
                                            w_sample)
                    sum = sum + val
                    num_sample = num_sample + 1
            if num_sample == 0:
                self.out[n_out][ctop][p_h][p_w] = 0
            else:
                self.out[n_out][ctop][p_h][p_w] = sum / num_sample
            self.top_count[n_out][ctop][p_h][p_w] = num_sample

    def setUp(self):
        self.op_type = "deformable_psroi_pooling"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Output')


class TestDeformablePSROIPoolOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            input1 = fluid.data(name="input1",
                                shape=[2, 192, 64, 64],
                                dtype='float32')
            rois1 = fluid.data(name="rois1",
                               shape=[-1, 4],
                               dtype='float32',
                               lod_level=1)
            trans1 = fluid.data(name="trans1",
                                shape=[2, 384, 64, 64],
                                dtype='float32')

            # The `input` must be Variable and the data type of `input` Tensor must be one of float32 and float64.
            def test_input_type():
                fluid.layers.deformable_roi_pooling(input=[3, 4],
                                                    rois=rois1,
                                                    trans=trans1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_input_type)

            def test_input_tensor_dtype():
                input2 = fluid.data(name="input2",
                                    shape=[2, 192, 64, 64],
                                    dtype='int32')
                fluid.layers.deformable_roi_pooling(input=input2,
                                                    rois=rois1,
                                                    trans=trans1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_input_tensor_dtype)

            # The `rois` must be Variable and the data type of `rois` Tensor must be one of float32 and float64.
            def test_rois_type():
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=2,
                                                    trans=trans1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_rois_type)

            def test_rois_tensor_dtype():
                rois2 = fluid.data(name="rois2",
                                   shape=[-1, 4],
                                   dtype='int32',
                                   lod_level=1)
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=rois2,
                                                    trans=trans1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_rois_tensor_dtype)

            # The `trans` must be Variable and the data type of `trans` Tensor must be one of float32 and float64.
            def test_trans_type():
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=rois1,
                                                    trans=[2],
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_trans_type)

            def test_trans_tensor_dtype():
                trans2 = fluid.data(name="trans2",
                                    shape=[2, 384, 64, 64],
                                    dtype='int32')
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=rois1,
                                                    trans=trans2,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_trans_tensor_dtype)

            # The `group_size` must be one of list and tuple.
            # Each element must be int.
            def test_group_size_type():
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=rois1,
                                                    trans=trans1,
                                                    group_size=1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=(8, 8),
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_group_size_type)

            # The `part_size` must be one of list, tuple and None.
            # Each element must be int.
            def test_part_size_type():
                fluid.layers.deformable_roi_pooling(input=input1,
                                                    rois=rois1,
                                                    trans=trans1,
                                                    pooled_height=8,
                                                    pooled_width=8,
                                                    part_size=8,
                                                    sample_per_part=4,
                                                    position_sensitive=True)

            self.assertRaises(TypeError, test_part_size_type)


if __name__ == '__main__':
    unittest.main()
