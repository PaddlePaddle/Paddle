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
# model: configs^tood^tood_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,method||__sub__,method||__sub__,method||__add__,method||__add__,api||paddle.tensor.manipulation.stack,method||astype,api||paddle.tensor.manipulation.stack,method||astype,method||reshape,method||reshape,api||paddle.tensor.creation.full,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,method||__truediv__,api||paddle.tensor.manipulation.split
import unittest

import numpy as np

import paddle


class SIR34(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
    ):
        var_277 = paddle.tensor.creation.arange(end=152)
        var_278 = var_277.__add__(0.5)
        var_279 = var_278.__mul__(8.0)
        var_280 = paddle.tensor.creation.arange(end=100)
        var_281 = var_280.__add__(0.5)
        var_282 = var_281.__mul__(8.0)
        out = paddle.tensor.creation.meshgrid(var_282, var_279)
        var_283 = out[0]
        var_284 = out[1]
        var_285 = var_284.__sub__(32.0)
        var_286 = var_283.__sub__(32.0)
        var_287 = var_284.__add__(32.0)
        var_288 = var_283.__add__(32.0)
        var_289 = paddle.tensor.manipulation.stack(
            [var_285, var_286, var_287, var_288], axis=-1
        )
        var_290 = var_289.astype('float32')
        var_291 = paddle.tensor.manipulation.stack([var_284, var_283], axis=-1)
        var_292 = var_291.astype('float32')
        var_293 = var_290.reshape([-1, 4])
        var_294 = var_292.reshape([-1, 2])
        var_295 = paddle.tensor.creation.full([15200, 1], 8.0, dtype='float32')
        var_296 = paddle.tensor.creation.arange(end=76)
        var_297 = var_296.__add__(0.5)
        var_298 = var_297.__mul__(16.0)
        var_299 = paddle.tensor.creation.arange(end=50)
        var_300 = var_299.__add__(0.5)
        var_301 = var_300.__mul__(16.0)
        out = paddle.tensor.creation.meshgrid(var_301, var_298)
        var_302 = out[0]
        var_303 = out[1]
        var_304 = var_303.__sub__(64.0)
        var_305 = var_302.__sub__(64.0)
        var_306 = var_303.__add__(64.0)
        var_307 = var_302.__add__(64.0)
        var_308 = paddle.tensor.manipulation.stack(
            [var_304, var_305, var_306, var_307], axis=-1
        )
        var_309 = var_308.astype('float32')
        var_310 = paddle.tensor.manipulation.stack([var_303, var_302], axis=-1)
        var_311 = var_310.astype('float32')
        var_312 = var_309.reshape([-1, 4])
        var_313 = var_311.reshape([-1, 2])
        var_314 = paddle.tensor.creation.full([3800, 1], 16.0, dtype='float32')
        var_315 = paddle.tensor.creation.arange(end=38)
        var_316 = var_315.__add__(0.5)
        var_317 = var_316.__mul__(32.0)
        var_318 = paddle.tensor.creation.arange(end=25)
        var_319 = var_318.__add__(0.5)
        var_320 = var_319.__mul__(32.0)
        out = paddle.tensor.creation.meshgrid(var_320, var_317)
        var_321 = out[0]
        var_322 = out[1]
        var_323 = var_322.__sub__(128.0)
        var_324 = var_321.__sub__(128.0)
        var_325 = var_322.__add__(128.0)
        var_326 = var_321.__add__(128.0)
        var_327 = paddle.tensor.manipulation.stack(
            [var_323, var_324, var_325, var_326], axis=-1
        )
        var_328 = var_327.astype('float32')
        var_329 = paddle.tensor.manipulation.stack([var_322, var_321], axis=-1)
        var_330 = var_329.astype('float32')
        var_331 = var_328.reshape([-1, 4])
        var_332 = var_330.reshape([-1, 2])
        var_333 = paddle.tensor.creation.full([950, 1], 32.0, dtype='float32')
        var_334 = paddle.tensor.creation.arange(end=19)
        var_335 = var_334.__add__(0.5)
        var_336 = var_335.__mul__(64.0)
        var_337 = paddle.tensor.creation.arange(end=13)
        var_338 = var_337.__add__(0.5)
        var_339 = var_338.__mul__(64.0)
        out = paddle.tensor.creation.meshgrid(var_339, var_336)
        var_340 = out[0]
        var_341 = out[1]
        var_342 = var_341.__sub__(256.0)
        var_343 = var_340.__sub__(256.0)
        var_344 = var_341.__add__(256.0)
        var_345 = var_340.__add__(256.0)
        var_346 = paddle.tensor.manipulation.stack(
            [var_342, var_343, var_344, var_345], axis=-1
        )
        var_347 = var_346.astype('float32')
        var_348 = paddle.tensor.manipulation.stack([var_341, var_340], axis=-1)
        var_349 = var_348.astype('float32')
        var_350 = var_347.reshape([-1, 4])
        var_351 = var_349.reshape([-1, 2])
        var_352 = paddle.tensor.creation.full([247, 1], 64.0, dtype='float32')
        var_353 = paddle.tensor.creation.arange(end=10)
        var_354 = var_353.__add__(0.5)
        var_355 = var_354.__mul__(128.0)
        var_356 = paddle.tensor.creation.arange(end=7)
        var_357 = var_356.__add__(0.5)
        var_358 = var_357.__mul__(128.0)
        out = paddle.tensor.creation.meshgrid(var_358, var_355)
        var_359 = out[0]
        var_360 = out[1]
        var_361 = var_360.__sub__(512.0)
        var_362 = var_359.__sub__(512.0)
        var_363 = var_360.__add__(512.0)
        var_364 = var_359.__add__(512.0)
        var_365 = paddle.tensor.manipulation.stack(
            [var_361, var_362, var_363, var_364], axis=-1
        )
        var_366 = var_365.astype('float32')
        var_367 = paddle.tensor.manipulation.stack([var_360, var_359], axis=-1)
        var_368 = var_367.astype('float32')
        var_369 = var_366.reshape([-1, 4])
        var_370 = var_368.reshape([-1, 2])
        var_371 = paddle.tensor.creation.full([70, 1], 128.0, dtype='float32')
        var_372 = paddle.tensor.manipulation.concat(
            [var_293, var_312, var_331, var_350, var_369]
        )
        var_373 = paddle.tensor.manipulation.concat(
            [var_294, var_313, var_332, var_351, var_370]
        )
        var_374 = paddle.tensor.manipulation.concat(
            [var_295, var_314, var_333, var_352, var_371]
        )
        var_375 = var_373.__truediv__(var_374)
        out = paddle.tensor.manipulation.split(
            var_375, [15200, 3800, 950, 247, 70]
        )
        var_376 = out[0]
        var_377 = out[1]
        var_378 = out[2]
        var_379 = out[3]
        var_380 = out[4]
        return (
            var_375,
            var_372,
            var_293,
            var_312,
            var_331,
            var_350,
            var_369,
            var_374,
        )


class TestSIR34(unittest.TestCase):
    def setUp(self):
        self.inputs = ()
        self.net = SIR34()

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
