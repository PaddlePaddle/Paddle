/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/tile_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    tile, ops::TileKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TileKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TileKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::TileKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TileKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TileKernel<paddle::platform::CUDADeviceContext, bool>);
REGISTER_OP_CUDA_KERNEL(
    tile_grad, ops::TileGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TileGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TileGradKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::TileGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TileGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
