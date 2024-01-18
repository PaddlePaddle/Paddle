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
# model: configs^rcnn_enhance^faster_rcnn_enhance_3x_coco_single_dy2st_train
# api||paddle.tensor.manipulation.split,api||paddle.tensor.manipulation.split,method||__add__,method||__truediv__,method||__add__,method||__truediv__,method||__sub__,method||__sub__,method||__add__,method||__truediv__,method||__add__,method||__truediv__,method||__sub__,method||__sub__,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,method||__sub__,method||__sub__,method||__mul__,api||paddle.tensor.logic.greater_than,method||__mul__,api||paddle.tensor.logic.greater_than,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__add__,method||__truediv__,method||__sub__,method||__sub__,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__sub__,method||__sub__,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||__add__,method||__add__,method||__add__,method||__truediv__,method||__truediv__,method||__truediv__,api||paddle.tensor.ops.atan,api||paddle.tensor.ops.atan,method||__sub__,method||__rmul__,method||__mul__,method||__rsub__,method||__add__,method||__add__,method||__truediv__,method||__mul__,method||__rsub__,method||__add__,method||__add__,method||__mul__,api||paddle.tensor.stat.mean,method||__mul__
import unittest

import numpy as np

import paddle


class SIR107(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1253,  # (shape: [32], dtype: paddle.float32, stop_gradient: False)
        var_1254,  # (shape: [32], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(
            var_1253, num_or_sections=4, axis=-1
        )
        var_1255 = out[0]
        var_1256 = out[1]
        var_1257 = out[2]
        var_1258 = out[3]
        out = paddle.tensor.manipulation.split(
            var_1254, num_or_sections=4, axis=-1
        )
        var_1259 = out[0]
        var_1260 = out[1]
        var_1261 = out[2]
        var_1262 = out[3]
        var_1263 = var_1255.__add__(var_1257)
        var_1264 = var_1263.__truediv__(2)
        var_1265 = var_1256.__add__(var_1258)
        var_1266 = var_1265.__truediv__(2)
        var_1267 = var_1257.__sub__(var_1255)
        var_1268 = var_1258.__sub__(var_1256)
        var_1269 = var_1259.__add__(var_1261)
        var_1270 = var_1269.__truediv__(2)
        var_1271 = var_1260.__add__(var_1262)
        var_1272 = var_1271.__truediv__(2)
        var_1273 = var_1261.__sub__(var_1259)
        var_1274 = var_1262.__sub__(var_1260)
        var_1275 = paddle.tensor.math.maximum(var_1255, var_1257)
        var_1276 = paddle.tensor.math.maximum(var_1256, var_1258)
        var_1277 = paddle.tensor.math.maximum(var_1255, var_1259)
        var_1278 = paddle.tensor.math.maximum(var_1256, var_1260)
        var_1279 = paddle.tensor.math.minimum(var_1275, var_1261)
        var_1280 = paddle.tensor.math.minimum(var_1276, var_1262)
        var_1281 = paddle.tensor.math.minimum(var_1255, var_1259)
        var_1282 = paddle.tensor.math.minimum(var_1256, var_1260)
        var_1283 = paddle.tensor.math.maximum(var_1275, var_1261)
        var_1284 = paddle.tensor.math.maximum(var_1276, var_1262)
        var_1285 = var_1279.__sub__(var_1277)
        var_1286 = var_1280.__sub__(var_1278)
        var_1287 = var_1285.__mul__(var_1286)
        var_1288 = paddle.tensor.logic.greater_than(var_1279, var_1277)
        var_1289 = var_1287.__mul__(var_1288)
        var_1290 = paddle.tensor.logic.greater_than(var_1280, var_1278)
        var_1291 = var_1289.__mul__(var_1290)
        var_1292 = var_1275.__sub__(var_1255)
        var_1293 = var_1276.__sub__(var_1256)
        var_1294 = var_1292.__mul__(var_1293)
        var_1295 = var_1261.__sub__(var_1259)
        var_1296 = var_1262.__sub__(var_1260)
        var_1297 = var_1295.__mul__(var_1296)
        var_1298 = var_1294.__add__(var_1297)
        var_1299 = var_1298.__sub__(var_1291)
        var_1300 = var_1299.__add__(1e-10)
        var_1301 = var_1291.__truediv__(var_1300)
        var_1302 = var_1264.__sub__(var_1270)
        var_1303 = var_1264.__sub__(var_1270)
        var_1304 = var_1302.__mul__(var_1303)
        var_1305 = var_1266.__sub__(var_1272)
        var_1306 = var_1266.__sub__(var_1272)
        var_1307 = var_1305.__mul__(var_1306)
        var_1308 = var_1304.__add__(var_1307)
        var_1309 = var_1283.__sub__(var_1281)
        var_1310 = var_1283.__sub__(var_1281)
        var_1311 = var_1309.__mul__(var_1310)
        var_1312 = var_1284.__sub__(var_1282)
        var_1313 = var_1284.__sub__(var_1282)
        var_1314 = var_1312.__mul__(var_1313)
        var_1315 = var_1311.__add__(var_1314)
        var_1316 = var_1308.__add__(1e-10)
        var_1317 = var_1315.__add__(1e-10)
        var_1318 = var_1316.__truediv__(var_1317)
        var_1319 = var_1273.__truediv__(var_1274)
        var_1320 = var_1267.__truediv__(var_1268)
        var_1321 = paddle.tensor.ops.atan(var_1319)
        var_1322 = paddle.tensor.ops.atan(var_1320)
        var_1323 = var_1321.__sub__(var_1322)
        var_1324 = var_1323.__rmul__(0.4052847345693511)
        var_1325 = var_1324.__mul__(var_1323)
        var_1326 = var_1301.__rsub__(1)
        var_1327 = var_1326.__add__(var_1325)
        var_1328 = var_1327.__add__(1e-10)
        var_1329 = var_1325.__truediv__(var_1328)
        var_1330 = var_1329.__mul__(var_1325)
        var_1331 = var_1301.__rsub__(1)
        var_1332 = var_1331.__add__(var_1330)
        var_1333 = var_1332.__add__(var_1318)
        var_1334 = var_1333.__mul__(1.0)
        var_1335 = paddle.tensor.stat.mean(var_1334)
        var_1336 = var_1335.__mul__(10.0)
        return var_1336


class TestSIR107(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[32], dtype=paddle.float32),
            paddle.rand(shape=[32], dtype=paddle.float32),
        )
        self.net = SIR107()

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
