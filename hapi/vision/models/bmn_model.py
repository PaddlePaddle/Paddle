#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
import numpy as np
import math

from hapi.model import Model, Loss
from hapi.download import get_weights_path

__all__ = ["BMN", "BmnLoss", "bmn"]

DATATYPE = 'float32'

pretrain_infos = {
    'bmn': ('https://paddlemodels.bj.bcebos.com/hapi/bmn.pdparams',
            '9286c821acc4cad46d6613b931ba468c')
}


def _get_interp1d_bin_mask(seg_xmin, seg_xmax, tscale, num_sample,
                           num_sample_perbin):
    """ generate sample mask for a boundary-matching pair """
    plen = float(seg_xmax - seg_xmin)
    plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
    total_samples = [
        seg_xmin + plen_sample * ii
        for ii in range(num_sample * num_sample_perbin)
    ]
    p_mask = []
    for idx in range(num_sample):
        bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) *
                                    num_sample_perbin]
        bin_vector = np.zeros([tscale])
        for sample in bin_samples:
            sample_upper = math.ceil(sample)
            sample_decimal, sample_down = math.modf(sample)
            if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                bin_vector[int(sample_down)] += 1 - sample_decimal
            if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                bin_vector[int(sample_upper)] += sample_decimal
        bin_vector = 1.0 / num_sample_perbin * bin_vector
        p_mask.append(bin_vector)
    p_mask = np.stack(p_mask, axis=1)
    return p_mask


def get_interp1d_mask(tscale, dscale, prop_boundary_ratio, num_sample,
                      num_sample_perbin):
    """ generate sample mask for each point in Boundary-Matching Map """
    mask_mat = []
    for start_index in range(tscale):
        mask_mat_vector = []
        for duration_index in range(dscale):
            if start_index + duration_index < tscale:
                p_xmin = start_index
                p_xmax = start_index + duration_index
                center_len = float(p_xmax - p_xmin) + 1
                sample_xmin = p_xmin - center_len * prop_boundary_ratio
                sample_xmax = p_xmax + center_len * prop_boundary_ratio
                p_mask = _get_interp1d_bin_mask(sample_xmin, sample_xmax,
                                                tscale, num_sample,
                                                num_sample_perbin)
            else:
                p_mask = np.zeros([tscale, num_sample])
            mask_mat_vector.append(p_mask)
        mask_mat_vector = np.stack(mask_mat_vector, axis=2)
        mask_mat.append(mask_mat_vector)
    mask_mat = np.stack(mask_mat, axis=3)
    mask_mat = mask_mat.astype(np.float32)

    sample_mask = np.reshape(mask_mat, [tscale, -1])
    return sample_mask


# Net
class Conv1D(fluid.dygraph.Layer):
    def __init__(self,
                 prefix,
                 num_channels=256,
                 num_filters=256,
                 size_k=3,
                 padding=1,
                 groups=1,
                 act="relu"):
        super(Conv1D, self).__init__()
        fan_in = num_channels * size_k * 1
        k = 1. / math.sqrt(fan_in)
        param_attr = ParamAttr(
            name=prefix + "_w",
            initializer=fluid.initializer.Uniform(
                low=-k, high=k))
        bias_attr = ParamAttr(
            name=prefix + "_b",
            initializer=fluid.initializer.Uniform(
                low=-k, high=k))

        self._conv2d = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=(1, size_k),
            stride=1,
            padding=(0, padding),
            groups=groups,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)

    def forward(self, x):
        x = fluid.layers.unsqueeze(input=x, axes=[2])
        x = self._conv2d(x)
        x = fluid.layers.squeeze(input=x, axes=[2])
        return x


class BMN(Model):
    """BMN model from
    `"BMN: Boundary-Matching Network for Temporal Action Proposal Generation" <https://arxiv.org/abs/1907.09702>`_

    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
        prop_boundary_ratio (float): ratio of expanded temporal region in proposal boundary, default 0.5. 
        num_sample (int): number of samples betweent starting boundary and ending boundary of each propoasl, default 32.
        num_sample_perbin (int):  number of selected points in each sample, default 3.
    """

    def __init__(self, tscale, dscale, prop_boundary_ratio, num_sample,
                 num_sample_perbin):
        super(BMN, self).__init__()

        #init config
        self.tscale = tscale
        self.dscale = dscale
        self.prop_boundary_ratio = prop_boundary_ratio
        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        # Base Module
        self.b_conv1 = Conv1D(
            prefix="Base_1",
            num_channels=400,
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.b_conv2 = Conv1D(
            prefix="Base_2",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")

        # Temporal Evaluation Module
        self.ts_conv1 = Conv1D(
            prefix="TEM_s1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.ts_conv2 = Conv1D(
            prefix="TEM_s2", num_filters=1, size_k=1, padding=0, act="sigmoid")
        self.te_conv1 = Conv1D(
            prefix="TEM_e1",
            num_filters=self.hidden_dim_1d,
            size_k=3,
            padding=1,
            groups=4,
            act="relu")
        self.te_conv2 = Conv1D(
            prefix="TEM_e2", num_filters=1, size_k=1, padding=0, act="sigmoid")

        #Proposal Evaluation Module
        self.p_conv1 = Conv1D(
            prefix="PEM_1d",
            num_filters=self.hidden_dim_2d,
            size_k=3,
            padding=1,
            act="relu")

        # get sample mask 
        sample_mask_array = get_interp1d_mask(
            self.tscale, self.dscale, self.prop_boundary_ratio,
            self.num_sample, self.num_sample_perbin)
        self.sample_mask = fluid.layers.create_parameter(
            shape=[self.tscale, self.num_sample * self.dscale * self.tscale],
            dtype=DATATYPE,
            attr=fluid.ParamAttr(
                name="sample_mask", trainable=False),
            default_initializer=fluid.initializer.NumpyArrayInitializer(
                sample_mask_array))

        self.sample_mask.stop_gradient = True

        self.p_conv3d1 = fluid.dygraph.Conv3D(
            num_channels=128,
            num_filters=self.hidden_dim_3d,
            filter_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
            padding=0,
            act="relu",
            param_attr=ParamAttr(name="PEM_3d1_w"),
            bias_attr=ParamAttr(name="PEM_3d1_b"))

        self.p_conv2d1 = fluid.dygraph.Conv2D(
            num_channels=512,
            num_filters=self.hidden_dim_2d,
            filter_size=1,
            stride=1,
            padding=0,
            act="relu",
            param_attr=ParamAttr(name="PEM_2d1_w"),
            bias_attr=ParamAttr(name="PEM_2d1_b"))
        self.p_conv2d2 = fluid.dygraph.Conv2D(
            num_channels=128,
            num_filters=self.hidden_dim_2d,
            filter_size=3,
            stride=1,
            padding=1,
            act="relu",
            param_attr=ParamAttr(name="PEM_2d2_w"),
            bias_attr=ParamAttr(name="PEM_2d2_b"))
        self.p_conv2d3 = fluid.dygraph.Conv2D(
            num_channels=128,
            num_filters=self.hidden_dim_2d,
            filter_size=3,
            stride=1,
            padding=1,
            act="relu",
            param_attr=ParamAttr(name="PEM_2d3_w"),
            bias_attr=ParamAttr(name="PEM_2d3_b"))
        self.p_conv2d4 = fluid.dygraph.Conv2D(
            num_channels=128,
            num_filters=2,
            filter_size=1,
            stride=1,
            padding=0,
            act="sigmoid",
            param_attr=ParamAttr(name="PEM_2d4_w"),
            bias_attr=ParamAttr(name="PEM_2d4_b"))

    def forward(self, x):
        #Base Module
        x = self.b_conv1(x)
        x = self.b_conv2(x)

        #TEM
        xs = self.ts_conv1(x)
        xs = self.ts_conv2(xs)
        xs = fluid.layers.squeeze(xs, axes=[1])
        xe = self.te_conv1(x)
        xe = self.te_conv2(xe)
        xe = fluid.layers.squeeze(xe, axes=[1])

        #PEM
        xp = self.p_conv1(x)
        #BM layer
        xp = fluid.layers.matmul(xp, self.sample_mask)
        xp = fluid.layers.reshape(
            xp, shape=[0, 0, -1, self.dscale, self.tscale])

        xp = self.p_conv3d1(xp)
        xp = fluid.layers.squeeze(xp, axes=[2])
        xp = self.p_conv2d1(xp)
        xp = self.p_conv2d2(xp)
        xp = self.p_conv2d3(xp)
        xp = self.p_conv2d4(xp)
        return xp, xs, xe


class BmnLoss(Loss):
    """Loss for BMN model

    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
    """

    def __init__(self, tscale, dscale):
        super(BmnLoss, self).__init__()
        self.tscale = tscale
        self.dscale = dscale

    def _get_mask(self):
        bm_mask = []
        for idx in range(self.dscale):
            mask_vector = [1 for i in range(self.tscale - idx)
                           ] + [0 for i in range(idx)]
            bm_mask.append(mask_vector)
        bm_mask = np.array(bm_mask, dtype=np.float32)
        self_bm_mask = fluid.layers.create_global_var(
            shape=[self.dscale, self.tscale],
            value=0,
            dtype=DATATYPE,
            persistable=True)
        fluid.layers.assign(bm_mask, self_bm_mask)
        self_bm_mask.stop_gradient = True
        return self_bm_mask

    def tem_loss_func(self, pred_start, pred_end, gt_start, gt_end):
        def bi_loss(pred_score, gt_label):
            pred_score = fluid.layers.reshape(
                x=pred_score, shape=[-1], inplace=False)
            gt_label = fluid.layers.reshape(
                x=gt_label, shape=[-1], inplace=False)
            gt_label.stop_gradient = True
            pmask = fluid.layers.cast(x=(gt_label > 0.5), dtype=DATATYPE)
            num_entries = fluid.layers.cast(
                fluid.layers.shape(pmask), dtype=DATATYPE)
            num_positive = fluid.layers.cast(
                fluid.layers.reduce_sum(pmask), dtype=DATATYPE)
            ratio = num_entries / num_positive
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio
            epsilon = 0.000001
            temp = fluid.layers.log(pred_score + epsilon)
            loss_pos = fluid.layers.elementwise_mul(
                fluid.layers.log(pred_score + epsilon), pmask)
            loss_pos = coef_1 * fluid.layers.reduce_mean(loss_pos)
            loss_neg = fluid.layers.elementwise_mul(
                fluid.layers.log(1.0 - pred_score + epsilon), (1.0 - pmask))
            loss_neg = coef_0 * fluid.layers.reduce_mean(loss_neg)
            loss = -1 * (loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss_func(self, pred_score, gt_iou_map, mask):

        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)

        u_hmask = fluid.layers.cast(x=gt_iou_map > 0.7, dtype=DATATYPE)
        u_mmask = fluid.layers.logical_and(gt_iou_map <= 0.7, gt_iou_map > 0.3)
        u_mmask = fluid.layers.cast(x=u_mmask, dtype=DATATYPE)
        u_lmask = fluid.layers.logical_and(gt_iou_map <= 0.3, gt_iou_map >= 0.)
        u_lmask = fluid.layers.cast(x=u_lmask, dtype=DATATYPE)
        u_lmask = fluid.layers.elementwise_mul(u_lmask, mask)

        num_h = fluid.layers.cast(
            fluid.layers.reduce_sum(u_hmask), dtype=DATATYPE)
        num_m = fluid.layers.cast(
            fluid.layers.reduce_sum(u_mmask), dtype=DATATYPE)
        num_l = fluid.layers.cast(
            fluid.layers.reduce_sum(u_lmask), dtype=DATATYPE)

        r_m = num_h / num_m
        u_smmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_smmask = fluid.layers.elementwise_mul(u_mmask, u_smmask)
        u_smmask = fluid.layers.cast(x=(u_smmask > (1. - r_m)), dtype=DATATYPE)

        r_l = num_h / num_l
        u_slmask = fluid.layers.uniform_random(
            shape=[gt_iou_map.shape[1], gt_iou_map.shape[2]],
            dtype=DATATYPE,
            min=0.0,
            max=1.0)
        u_slmask = fluid.layers.elementwise_mul(u_lmask, u_slmask)
        u_slmask = fluid.layers.cast(x=(u_slmask > (1. - r_l)), dtype=DATATYPE)

        weights = u_hmask + u_smmask + u_slmask
        weights.stop_gradient = True
        loss = fluid.layers.square_error_cost(pred_score, gt_iou_map)
        loss = fluid.layers.elementwise_mul(loss, weights)
        loss = 0.5 * fluid.layers.reduce_sum(loss) / fluid.layers.reduce_sum(
            weights)

        return loss

    def pem_cls_loss_func(self, pred_score, gt_iou_map, mask):
        gt_iou_map = fluid.layers.elementwise_mul(gt_iou_map, mask)
        gt_iou_map.stop_gradient = True
        pmask = fluid.layers.cast(x=(gt_iou_map > 0.9), dtype=DATATYPE)
        nmask = fluid.layers.cast(x=(gt_iou_map <= 0.9), dtype=DATATYPE)
        nmask = fluid.layers.elementwise_mul(nmask, mask)

        num_positive = fluid.layers.reduce_sum(pmask)
        num_entries = num_positive + fluid.layers.reduce_sum(nmask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        epsilon = 0.000001
        loss_pos = fluid.layers.elementwise_mul(
            fluid.layers.log(pred_score + epsilon), pmask)
        loss_pos = coef_1 * fluid.layers.reduce_sum(loss_pos)
        loss_neg = fluid.layers.elementwise_mul(
            fluid.layers.log(1.0 - pred_score + epsilon), nmask)
        loss_neg = coef_0 * fluid.layers.reduce_sum(loss_neg)
        loss = -1 * (loss_pos + loss_neg) / num_entries
        return loss

    def forward(self, outputs, labels):
        pred_bm, pred_start, pred_end = outputs
        if len(labels) == 3:
            gt_iou_map, gt_start, gt_end = labels
        elif len(labels) == 4:  # video_index used in eval mode
            gt_iou_map, gt_start, gt_end, video_index = labels
        pred_bm_reg = fluid.layers.squeeze(
            fluid.layers.slice(
                pred_bm, axes=[1], starts=[0], ends=[1]),
            axes=[1])
        pred_bm_cls = fluid.layers.squeeze(
            fluid.layers.slice(
                pred_bm, axes=[1], starts=[1], ends=[2]),
            axes=[1])

        bm_mask = self._get_mask()

        pem_reg_loss = self.pem_reg_loss_func(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss_func(pred_bm_cls, gt_iou_map, bm_mask)

        tem_loss = self.tem_loss_func(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10 * pem_reg_loss + pem_cls_loss
        return loss


def bmn(tscale,
        dscale,
        prop_boundary_ratio,
        num_sample,
        num_sample_perbin,
        pretrained=True):
    """BMN model
    
    Args:
        tscale (int): sequence length, default 100.
        dscale (int): max duration length, default 100.
        prop_boundary_ratio (float): ratio of expanded temporal region in proposal boundary, default 0.5. 
        num_sample (int): number of samples betweent starting boundary and ending boundary of each propoasl, default 32.
        num_sample_perbin (int):  number of selected points in each sample, default 3.
        pretrained (bool): If True, returns a model with pre-trained model, default True.
    """
    model = BMN(tscale, dscale, prop_boundary_ratio, num_sample,
                num_sample_perbin)
    if pretrained:
        weight_path = get_weights_path(*(pretrain_infos['bmn']))
        assert weight_path.endswith('.pdparams'), \
                "suffix of weight must be .pdparams"
        model.load(weight_path[:-9])
    return model
