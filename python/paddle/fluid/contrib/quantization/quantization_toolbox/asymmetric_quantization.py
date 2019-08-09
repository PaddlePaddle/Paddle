# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# Co-authored-by: Huan Jun, Li Xingjian, Xiong Haoyi, Yang Yingzhen, Zhao Baoxin
# Acknowledge: Thanks for the help of Lu jun.
import numpy as np
import paddle.fluid as fluid


class asymmetric_quantization():
    def __init__(self, quant_level=256):
        self.quant_level = quant_level

    def quantization(self, net, symmetric=False):
        ori_dict = net.copy()
        quant_level = self.quant_level  # 255
        for name, param in ori_dict.items():
            if 'Conv2D' in name or 'FC' in name:
                ori_value = np.array(param._ivar.value().get_tensor())
                fm = np.array(param._ivar.value().get_tensor())
                fm_max_ori = fm.max()
                fm_min_ori = fm.min()

                if not symmetric:
                    ss = (fm_max_ori - fm_min_ori) / (quant_level - 1)
                    m0 = np.round((fm_min_ori / ss))
                    fm_min = ss * m0
                    fm_diff = fm - fm_min
                    fm_diff_ind = np.round(fm_diff / ss)
                    fm_diff_ind = np.maximum(fm_diff_ind, 0)
                    fm_diff_ind = np.minimum(fm_diff_ind, 255)
                    fm_diff_ind = fm_diff_ind - 128  #
                    m0 = m0 + 128
                    if m0 < -128 or m0 > 127:
                        print('m0 = %d out of range\n' % (m0))
                        if m0 < -128:
                            m0 = -128
                        if m0 > 127:
                            m0 = 127
                    fm_diff_ind_16 = np.int16(fm_diff_ind)
                    m0_16 = np.int16(m0)
                    ind_m0 = fm_diff_ind_16 + m0_16
                    ind_m0_f = ind_m0.astype(np.float)
                    item_var = param._ivar.value()
                    tensor = item_var.get_tensor()
                    changed_np = ind_m0_f * ss
                    tensor.set(
                        changed_np.astype(np.float32),
                        fluid.framework._current_expected_place())
                    changed_value = np.array(param._ivar.value().get_tensor())
                    delta_value = ori_value - changed_value
                    # if 'FC' in name:
                    #     print(name, fm.sum(), changed_np.sum(), changed_value.sum(), delta_value.sum())
        return ori_dict
