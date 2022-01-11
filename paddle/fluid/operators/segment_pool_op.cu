/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/segment_pool_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    segment_pool,
    ops::SegmentPoolKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SegmentPoolKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    segment_pool_grad,
    ops::SegmentPoolGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SegmentPoolGradKernel<paddle::platform::CUDADeviceContext, double>);
