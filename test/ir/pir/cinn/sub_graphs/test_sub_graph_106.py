# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# repo: PaddleDetection
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# method||astype,method||unsqueeze,method||tile,method||astype,api||paddle.tensor.search.masked_select,method||reshape,api||paddle.tensor.search.masked_select,method||reshape,method||sum,api||paddle.tensor.search.masked_select,method||unsqueeze,api||paddle.nn.functional.loss.l1_loss,api||paddle.tensor.manipulation.split,api||paddle.tensor.manipulation.split,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,method||__sub__,method||clip,method||__sub__,method||clip,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__add__,method||__truediv__,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__truediv__,method||__sub__,method||__rsub__,method||__mul__,method||__mul__,method||sum,method||__truediv__,method||unsqueeze,method||astype,method||tile,method||astype,api||paddle.tensor.search.masked_select,method||reshape,api||paddle.tensor.manipulation.split,method||__sub__,method||__sub__,api||paddle.tensor.manipulation.concat,method||clip,api||paddle.tensor.search.masked_select,method||reshape,method||floor,api||paddle.tensor.manipulation.cast,method||__add__,method||astype,method||__sub__,method||__rsub__,method||__sub__,api||paddle.nn.functional.loss.cross_entropy,method||__mul__,method||__sub__,api||paddle.nn.functional.loss.cross_entropy,method||__mul__,method||__add__,method||mean,method||__mul__,method||sum,method||__truediv__
import unittest

import numpy as np

import paddle


class SIR179(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1152,  # (shape: [1, 4116], dtype: paddle.bool, stop_gradient: True)
        var_1153,  # (shape: [1, 4116, 4], dtype: paddle.float32, stop_gradient: False)
        var_1154,  # (shape: [1, 4116, 4], dtype: paddle.float32, stop_gradient: True)
        var_1155,  # (shape: [1, 4116, 80], dtype: paddle.float32, stop_gradient: True)
        var_1156,  # (shape: [], dtype: paddle.float32, stop_gradient: True)
        var_1157,  # (shape: [1, 4116, 68], dtype: paddle.float32, stop_gradient: False)
        var_1158,  # (shape: [4116, 2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1159 = var_1152.astype('int32')
        var_1160 = var_1159.unsqueeze(-1)
        var_1161 = var_1160.tile([1, 1, 4])
        var_1162 = var_1161.astype('bool')
        var_1163 = paddle.tensor.search.masked_select(var_1153, var_1162)
        var_1164 = var_1163.reshape([-1, 4])
        var_1165 = paddle.tensor.search.masked_select(var_1154, var_1162)
        var_1166 = var_1165.reshape([-1, 4])
        var_1167 = var_1155.sum(-1)
        var_1168 = paddle.tensor.search.masked_select(var_1167, var_1152)
        var_1169 = var_1168.unsqueeze(-1)
        var_1170 = paddle.nn.functional.loss.l1_loss(var_1164, var_1166)
        out = paddle.tensor.manipulation.split(
            var_1164, num_or_sections=4, axis=-1
        )
        var_1171 = out[0]
        var_1172 = out[1]
        var_1173 = out[2]
        var_1174 = out[3]
        out = paddle.tensor.manipulation.split(
            var_1166, num_or_sections=4, axis=-1
        )
        var_1175 = out[0]
        var_1176 = out[1]
        var_1177 = out[2]
        var_1178 = out[3]
        var_1179 = paddle.tensor.math.maximum(var_1171, var_1175)
        var_1180 = paddle.tensor.math.maximum(var_1172, var_1176)
        var_1181 = paddle.tensor.math.minimum(var_1173, var_1177)
        var_1182 = paddle.tensor.math.minimum(var_1174, var_1178)
        var_1183 = var_1181.__sub__(var_1179)
        var_1184 = var_1183.clip(0)
        var_1185 = var_1182.__sub__(var_1180)
        var_1186 = var_1185.clip(0)
        var_1187 = var_1184.__mul__(var_1186)
        var_1188 = var_1173.__sub__(var_1171)
        var_1189 = var_1174.__sub__(var_1172)
        var_1190 = var_1188.__mul__(var_1189)
        var_1191 = var_1177.__sub__(var_1175)
        var_1192 = var_1178.__sub__(var_1176)
        var_1193 = var_1191.__mul__(var_1192)
        var_1194 = var_1190.__add__(var_1193)
        var_1195 = var_1194.__sub__(var_1187)
        var_1196 = var_1195.__add__(1e-10)
        var_1197 = var_1187.__truediv__(var_1196)
        var_1198 = paddle.tensor.math.minimum(var_1171, var_1175)
        var_1199 = paddle.tensor.math.minimum(var_1172, var_1176)
        var_1200 = paddle.tensor.math.maximum(var_1173, var_1177)
        var_1201 = paddle.tensor.math.maximum(var_1174, var_1178)
        var_1202 = var_1200.__sub__(var_1198)
        var_1203 = var_1201.__sub__(var_1199)
        var_1204 = var_1202.__mul__(var_1203)
        var_1205 = var_1204.__add__(1e-10)
        var_1206 = var_1205.__sub__(var_1196)
        var_1207 = var_1206.__truediv__(var_1205)
        var_1208 = var_1197.__sub__(var_1207)
        var_1209 = var_1208.__rsub__(1)
        var_1210 = var_1209.__mul__(1.0)
        var_1211 = var_1210.__mul__(var_1169)
        var_1212 = var_1211.sum()
        var_1213 = var_1212.__truediv__(var_1156)
        var_1214 = var_1152.unsqueeze(-1)
        var_1215 = var_1214.astype('int32')
        var_1216 = var_1215.tile([1, 1, 68])
        var_1217 = var_1216.astype('bool')
        var_1218 = paddle.tensor.search.masked_select(var_1157, var_1217)
        var_1219 = var_1218.reshape([-1, 4, 17])
        out = paddle.tensor.manipulation.split(var_1154, 2, -1)
        var_1220 = out[0]
        var_1221 = out[1]
        var_1222 = var_1158.__sub__(var_1220)
        var_1223 = var_1221.__sub__(var_1158)
        var_1224 = paddle.tensor.manipulation.concat([var_1222, var_1223], -1)
        var_1225 = var_1224.clip(0, 15.99)
        var_1226 = paddle.tensor.search.masked_select(var_1225, var_1162)
        var_1227 = var_1226.reshape([-1, 4])
        var_1228 = var_1227.floor()
        var_1229 = paddle.tensor.manipulation.cast(var_1228, 'int64')
        var_1230 = var_1229.__add__(1)
        var_1231 = var_1230.astype('float32')
        var_1232 = var_1231.__sub__(var_1227)
        var_1233 = var_1232.__rsub__(1)
        var_1234 = var_1229.__sub__(0)
        var_1235 = paddle.nn.functional.loss.cross_entropy(
            var_1219, var_1234, reduction='none'
        )
        var_1236 = var_1235.__mul__(var_1232)
        var_1237 = var_1230.__sub__(0)
        var_1238 = paddle.nn.functional.loss.cross_entropy(
            var_1219, var_1237, reduction='none'
        )
        var_1239 = var_1238.__mul__(var_1233)
        var_1240 = var_1236.__add__(var_1239)
        var_1241 = var_1240.mean(-1, keepdim=True)
        var_1242 = var_1241.__mul__(var_1169)
        var_1243 = var_1242.sum()
        var_1244 = var_1243.__truediv__(var_1156)
        return var_1170, var_1213, var_1244


class TestSIR179(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 4116], dtype=paddle.bool),
            paddle.rand(shape=[1, 4116, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 4116, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 4116, 80], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1, 4116, 68], dtype=paddle.float32),
            paddle.rand(shape=[4116, 2], dtype=paddle.float32),
        )
        self.net = SIR179()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
