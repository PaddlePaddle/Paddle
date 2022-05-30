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

#include "paddle/fluid/operators/fake_dequantize_op.cu.h"
#include "paddle/fluid/operators/fake_dequantize_op.h"

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
using float16 = paddle::platform::float16;
REGISTER_OP_CUDA_KERNEL(fake_dequantize_max_abs,
                        ops::FakeDequantizeMaxAbsKernel<CUDA, float>,
                        ops::FakeDequantizeMaxAbsKernel<CUDA, double>,
                        ops::FakeDequantizeMaxAbsKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_channel_wise_dequantize_max_abs,
    ops::FakeChannelWiseDequantizeMaxAbsKernel<CUDA, float>,
    ops::FakeChannelWiseDequantizeMaxAbsKernel<CUDA, double>,
    ops::FakeChannelWiseDequantizeMaxAbsKernel<CUDA, float16>);
