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

#include "paddle/fluid/operators/fake_quantize_op.cu.h"
#include "paddle/fluid/operators/fake_quantize_op.h"

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
using float16 = paddle::platform::float16;
REGISTER_OP_CUDA_KERNEL(fake_quantize_abs_max,
                        ops::FakeQuantizeAbsMaxKernel<CUDA, float>,
                        ops::FakeQuantizeAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(fake_quantize_dequantize_abs_max,
                        ops::FakeQuantizeDequantizeAbsMaxKernel<CUDA, float>,
                        ops::FakeQuantizeDequantizeAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_channel_wise_quantize_abs_max,
    ops::FakeChannelWiseQuantizeAbsMaxKernel<CUDA, float>,
    ops::FakeChannelWiseQuantizeAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(fake_quantize_range_abs_max,
                        ops::FakeQuantizeRangeAbsMaxKernel<CUDA, float>,
                        ops::FakeQuantizeRangeAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_quantize_moving_average_abs_max,
    ops::FakeQuantizeMovingAverageAbsMaxKernel<CUDA, float>,
    ops::FakeQuantizeMovingAverageAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(moving_average_abs_max_scale,
                        ops::MovingAverageAbsMaxScaleKernel<CUDA, float>,
                        ops::MovingAverageAbsMaxScaleKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_quantize_dequantize_moving_average_abs_max,
    ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel<CUDA, float>,
    ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(stright_throuth_estimator_grad,
                        ops::StrightThroughEstimatorGradKernel<CUDA, float>,
                        ops::StrightThroughEstimatorGradKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_channel_wise_quantize_dequantize_abs_max,
    ops::FakeChannelWiseQuantizeDequantizeAbsMaxKernel<CUDA, float>);
