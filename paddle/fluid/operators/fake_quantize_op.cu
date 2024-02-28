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

#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/operators/fake_quantize_op.cu.h"

namespace ops = paddle::operators;
using float16 = paddle::platform::float16;

PD_REGISTER_STRUCT_KERNEL(fake_quantize_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_quantize_dequantize_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeDequantizeAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_channel_wise_quantize_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeChannelWiseQuantizeAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_quantize_range_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeRangeAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_quantize_moving_average_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeMovingAverageAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(moving_average_abs_max_scale,
                          GPU,
                          ALL_LAYOUT,
                          ops::MovingAverageAbsMaxScaleKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_quantize_dequantize_moving_average_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(straight_through_estimator_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::StraightThroughEstimatorGradKernel,
                          float,
                          float16) {}
PD_REGISTER_STRUCT_KERNEL(fake_channel_wise_quantize_dequantize_abs_max,
                          GPU,
                          ALL_LAYOUT,
                          ops::FakeChannelWiseQuantizeDequantizeAbsMaxKernel,
                          float) {}
