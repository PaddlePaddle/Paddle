from __future__ import print_function
import math
import numpy as np
import unittest
import paddle.compat as cpt
from op_test import OpTest


class TestDeformablePSROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.calc_deformable_psroi_pooling()
        self.inputs = {'Input': self.input, "ROIs": (self.rois[:, 1:5], 
                       self.rois_lod), "Trans": self.trans}
        self.attrs = {
            'no_trans': self.no_trans,
            'spatial_scale': self.spatial_scale,
            'output_dim': self.output_channels,
            'group_size': self.group_size,
            'pooled_height': self.pooled_height, 
            'pooled_width': self.pooled_width,
            'part_size': self.part_size,
            'sample_per_part': self.sample_per_part,
            'trans_std': self.trans_std
        }

        self.outputs = {'Output': self.out.astype('float32'),
                        'TopCount': self.top_count.astype('float32')}

    def init_test_case(self):
        self.batch_size = 3
        self.channels = 3 * 2 * 2
        self.height = 12
        self.width = 12
        self.input_dim = [self.batch_size, self.channels, self.height, self.width]
        self.no_trans = 1
        self.spatial_scale = 1.0 / 4.0
        self.output_channels = 12
        self.group_size = [1, 1]
        self.pooled_height = 4
        self.pooled_width = 4
        #self.pooled_size=4
        self.part_size = [4, 4]
        self.sample_per_part = 2
        self.trans_std = 0.1
        self.input = np.random.random(self.input_dim).astype('float32')

    def make_rois(self):
        rois = []
        self.rois_lod = [[]]
        for bno in range(self.batch_size):
            self.rois_lod[0].append(bno + 1)
            for i in range(bno + 1):
                x1 = np.random.random_integers(
                    0, self.width // self.spatial_scale - self.pooled_width)
                y1 = np.random.random_integers(
                    0, self.height // self.spatial_scale - self.pooled_height)

                x2 = np.random.random_integers(x1 + self.pooled_width,
                                               self.width // self.spatial_scale)
                y2 = np.random.random_integers(
                    y1 + self.pooled_height, self.height // self.spatial_scale)
                roi = [bno, x1, y1, x2, y2]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float32")

    def dmc_bilinear(self, data_im, h, w):
        h_low = int(np.floor(h))
        w_low = int(np.floor(w))
        h_high = h_low + 1
        w_high = w_low + 1
        lh = h - h_low
        lw = w - w_low
        hh = 1 - lh
        hw = 1 - lw
        v1 = 0
        if h_low >= 0 and w_low >= 0:
            v1 = data_im[h_low, w_low]
        v2 = 0
        if h_low >= 0 and w_high <= self.width - 1:
            v2 = data_im[h_low, w_high]
        v3 = 0
        if h_high <= self.height - 1 and w_low >= 0:
            v3 = data_im[h_high, w_low]
        v4 = 0
        if h_high <= self.height - 1 and w_high <= self.width - 1:
            v4 = data_im[h_high, w_high]
        w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw
        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
        return val
    
    def calc_deformable_psroi_pooling(self):
        output_shape = (self.rois_num, self.output_channels, self.pooled_height, self.pooled_width)

        self.out = np.zeros(output_shape)
        top_count = np.zeros(output_shape)
        self.trans = np.random.rand(self.rois_num, 2, self.part_size[0], 
                                    self.part_size[1]).astype('float32')
        self.top_count = np.random.random((output_shape)).astype('float32')
        count = self.rois_num * self.output_channels * self.pooled_height * self.pooled_width
        for index in range(count):
            pw = int(index % self.pooled_width)
            ph = int(index / self.pooled_width % self.pooled_height)
            ctop = int(index / self.pooled_width / self.pooled_height % self.output_channels)
            n = int(index / self.pooled_width / self.pooled_height / self.output_channels)

            roi = self.rois[n]
            roi_batch_id = int(roi[0])
            roi_start_w = int(np.round(roi[1])) * self.spatial_scale - 0.5
            roi_start_h = int(np.round(roi[2])) * self.spatial_scale - 0.5
            roi_end_w = int(np.round(roi[3]+1)) * self.spatial_scale - 0.5
            roi_end_h = int(np.round(roi[4]+1)) * self.spatial_scale - 0.5

            roi_width = max(roi_end_w - roi_start_w, 0.1)
            roi_height = max(roi_end_h - roi_start_h, 0.1)

            bin_size_h = float(roi_height) / float(self.pooled_height);
            bin_size_w = float(roi_width) / float(self.pooled_width);

            sub_bin_size_h = bin_size_h / self.sample_per_part;
            sub_bin_size_w = bin_size_w / self.sample_per_part;

            part_h = int(np.floor(ph) / self.pooled_height * self.part_size[0])
            part_w = int(np.floor(pw) / self.pooled_width * self.part_size[1])

            if self.no_trans:
                trans_x = 0
                trans_y = 0
            else:
                trans_x = self.trans[n][0][part_h][part_w] * self.trans_std
                trans_y = self.trans[n][1][part_h][part_w] * self.trans_std

            wstart = pw * bin_size_w + roi_start_w
            wstart = wstart + trans_x * roi_width
            hstart = ph * bin_size_h + roi_start_h
            hstart = hstart + trans_y * roi_height

            sum = 0
            count_time = 0
            gw = np.floor(pw * self.group_size[0] / self.pooled_height)
            gh = np.floor(ph * self.group_size[1] / self.pooled_width)
            gw = min(max(gw, 0), self.group_size[0] - 1)
            gh = min(max(gh, 0), self.group_size[1] - 1)
            #print(input[n, 1])
            input_i = self.input[roi_batch_id]
            for iw in range(self.sample_per_part):
                for ih in range(self.sample_per_part):
                    w = wstart + iw * sub_bin_size_w
                    h = hstart + ih * sub_bin_size_h
                    if w < -0.5 or w > self.width - 0.5 or \
                    h < -0.5 or h > self.height - 0.5:
                        continue
                    w = min(max(w, 0.), self.width - 1.)
                    h = min(max(h, 0.), self.height - 1.)
                    c = int((ctop * self.group_size[0] + gh) * self.group_size[1] + gw)
                    val = self.dmc_bilinear(input_i[c], h, w)
                    sum = sum + val
                    count_time = count_time + 1
            if count_time == 0 :
                self.out[n][c][ph][pw] = 0
            else:
                self.out[n][c][ph][pw] = sum / count_time
            self.top_count[n][c][ph][pw] = count_time
            
    def setUp(self):
        self.op_type = "deformable_psroi_pooling"
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Output')

if __name__ == '__main__':
    unittest.main()
