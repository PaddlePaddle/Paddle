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
# model: configs^rotate^fcosr^fcosr_x50_3x_dota_single_dy2st_train
# api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.arange,method||__add__,method||__mul__,api||paddle.tensor.creation.meshgrid,api||paddle.tensor.manipulation.stack,api||paddle.tensor.manipulation.cast,method||reshape,method||__mul__,api||paddle.tensor.creation.full,method||__mul__,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR35(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_284,  # (shape: [1, 256, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_285,  # (shape: [1, 256, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_286,  # (shape: [1, 256, 32, 32], dtype: paddle.float32, stop_gradient: False)
        var_287,  # (shape: [1, 256, 16, 16], dtype: paddle.float32, stop_gradient: False)
        var_288,  # (shape: [1, 256, 8, 8], dtype: paddle.float32, stop_gradient: False)
    ):
        var_289 = paddle.tensor.attribute.shape(var_284)
        var_290 = var_289.__getitem__(0)
        var_291 = var_289.__getitem__(1)
        var_292 = var_289.__getitem__(2)
        var_293 = var_289.__getitem__(3)
        var_294 = paddle.tensor.creation.arange(end=var_293)
        var_295 = var_294.__add__(0.5)
        var_296 = var_295.__mul__(8)
        var_297 = paddle.tensor.creation.arange(end=var_292)
        var_298 = var_297.__add__(0.5)
        var_299 = var_298.__mul__(8)
        out = paddle.tensor.creation.meshgrid(var_299, var_296)
        var_300 = out[0]
        var_301 = out[1]
        var_302 = paddle.tensor.manipulation.stack([var_301, var_300], axis=-1)
        var_303 = paddle.tensor.manipulation.cast(var_302, dtype='float32')
        var_304 = var_303.reshape([1, -1, 2])
        var_305 = var_292.__mul__(var_293)
        var_306 = paddle.tensor.creation.full(
            [1, var_305, 1], 8, dtype='float32'
        )
        var_307 = var_292.__mul__(var_293)
        var_308 = paddle.tensor.attribute.shape(var_285)
        var_309 = var_308.__getitem__(0)
        var_310 = var_308.__getitem__(1)
        var_311 = var_308.__getitem__(2)
        var_312 = var_308.__getitem__(3)
        var_313 = paddle.tensor.creation.arange(end=var_312)
        var_314 = var_313.__add__(0.5)
        var_315 = var_314.__mul__(16)
        var_316 = paddle.tensor.creation.arange(end=var_311)
        var_317 = var_316.__add__(0.5)
        var_318 = var_317.__mul__(16)
        out = paddle.tensor.creation.meshgrid(var_318, var_315)
        var_319 = out[0]
        var_320 = out[1]
        var_321 = paddle.tensor.manipulation.stack([var_320, var_319], axis=-1)
        var_322 = paddle.tensor.manipulation.cast(var_321, dtype='float32')
        var_323 = var_322.reshape([1, -1, 2])
        var_324 = var_311.__mul__(var_312)
        var_325 = paddle.tensor.creation.full(
            [1, var_324, 1], 16, dtype='float32'
        )
        var_326 = var_311.__mul__(var_312)
        var_327 = paddle.tensor.attribute.shape(var_286)
        var_328 = var_327.__getitem__(0)
        var_329 = var_327.__getitem__(1)
        var_330 = var_327.__getitem__(2)
        var_331 = var_327.__getitem__(3)
        var_332 = paddle.tensor.creation.arange(end=var_331)
        var_333 = var_332.__add__(0.5)
        var_334 = var_333.__mul__(32)
        var_335 = paddle.tensor.creation.arange(end=var_330)
        var_336 = var_335.__add__(0.5)
        var_337 = var_336.__mul__(32)
        out = paddle.tensor.creation.meshgrid(var_337, var_334)
        var_338 = out[0]
        var_339 = out[1]
        var_340 = paddle.tensor.manipulation.stack([var_339, var_338], axis=-1)
        var_341 = paddle.tensor.manipulation.cast(var_340, dtype='float32')
        var_342 = var_341.reshape([1, -1, 2])
        var_343 = var_330.__mul__(var_331)
        var_344 = paddle.tensor.creation.full(
            [1, var_343, 1], 32, dtype='float32'
        )
        var_345 = var_330.__mul__(var_331)
        var_346 = paddle.tensor.attribute.shape(var_287)
        var_347 = var_346.__getitem__(0)
        var_348 = var_346.__getitem__(1)
        var_349 = var_346.__getitem__(2)
        var_350 = var_346.__getitem__(3)
        var_351 = paddle.tensor.creation.arange(end=var_350)
        var_352 = var_351.__add__(0.5)
        var_353 = var_352.__mul__(64)
        var_354 = paddle.tensor.creation.arange(end=var_349)
        var_355 = var_354.__add__(0.5)
        var_356 = var_355.__mul__(64)
        out = paddle.tensor.creation.meshgrid(var_356, var_353)
        var_357 = out[0]
        var_358 = out[1]
        var_359 = paddle.tensor.manipulation.stack([var_358, var_357], axis=-1)
        var_360 = paddle.tensor.manipulation.cast(var_359, dtype='float32')
        var_361 = var_360.reshape([1, -1, 2])
        var_362 = var_349.__mul__(var_350)
        var_363 = paddle.tensor.creation.full(
            [1, var_362, 1], 64, dtype='float32'
        )
        var_364 = var_349.__mul__(var_350)
        var_365 = paddle.tensor.attribute.shape(var_288)
        var_366 = var_365.__getitem__(0)
        var_367 = var_365.__getitem__(1)
        var_368 = var_365.__getitem__(2)
        var_369 = var_365.__getitem__(3)
        var_370 = paddle.tensor.creation.arange(end=var_369)
        var_371 = var_370.__add__(0.5)
        var_372 = var_371.__mul__(128)
        var_373 = paddle.tensor.creation.arange(end=var_368)
        var_374 = var_373.__add__(0.5)
        var_375 = var_374.__mul__(128)
        out = paddle.tensor.creation.meshgrid(var_375, var_372)
        var_376 = out[0]
        var_377 = out[1]
        var_378 = paddle.tensor.manipulation.stack([var_377, var_376], axis=-1)
        var_379 = paddle.tensor.manipulation.cast(var_378, dtype='float32')
        var_380 = var_379.reshape([1, -1, 2])
        var_381 = var_368.__mul__(var_369)
        var_382 = paddle.tensor.creation.full(
            [1, var_381, 1], 128, dtype='float32'
        )
        var_383 = var_368.__mul__(var_369)
        var_384 = paddle.tensor.manipulation.concat(
            [var_304, var_323, var_342, var_361, var_380], axis=1
        )
        var_385 = paddle.tensor.manipulation.concat(
            [var_306, var_325, var_344, var_363, var_382], axis=1
        )
        return var_384, var_385, var_307, var_326, var_345, var_364, var_383


class TestSIR35(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 128, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 32, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 16, 16], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 8, 8], dtype=paddle.float32),
        )
        self.net = SIR35()

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
