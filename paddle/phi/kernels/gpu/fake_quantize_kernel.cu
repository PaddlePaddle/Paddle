// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/fake_quantize_kernel_impl.h"

PD_REGISTER_KERNEL(fake_quantize_range_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeRangeAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fake_quantize_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fake_quantize_moving_average_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantOrWithDequantMovingAverageAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fake_channel_wise_quantize_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeChannelWiseQuantizeAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fake_channel_wise_quantize_dequantize_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeChannelWiseQuantizeDequantizeAbsMaxKernel,
                   float) {}

PD_REGISTER_KERNEL(fake_quantize_dequantize_moving_average_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeMovingAverageAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}

PD_REGISTER_KERNEL(fake_quantize_dequantize_abs_max,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeQuantizeDequantizeAbsMaxKernel,
                   float,
                   phi::dtype::float16) {}
