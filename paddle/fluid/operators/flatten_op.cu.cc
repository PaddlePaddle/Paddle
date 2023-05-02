/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/flatten_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(flatten,
                        ops::FlattenKernel<phi::GPUContext, float>,
                        ops::FlattenKernel<phi::GPUContext, double>,
                        ops::FlattenKernel<phi::GPUContext, uint8_t>,
                        ops::FlattenKernel<phi::GPUContext, int>,
                        ops::FlattenKernel<phi::GPUContext, int8_t>,
                        ops::FlattenKernel<phi::GPUContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(flatten_grad,
                        ops::FlattenGradKernel<phi::GPUContext, float>,
                        ops::FlattenGradKernel<phi::GPUContext, double>,
                        ops::FlattenGradKernel<phi::GPUContext, uint8_t>,
                        ops::FlattenGradKernel<phi::GPUContext, int>,
                        ops::FlattenGradKernel<phi::GPUContext, int8_t>,
                        ops::FlattenGradKernel<phi::GPUContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(flatten2,
                        ops::Flatten2Kernel<phi::GPUContext, float>,
                        ops::Flatten2Kernel<phi::GPUContext, double>,
                        ops::Flatten2Kernel<phi::GPUContext, uint8_t>,
                        ops::Flatten2Kernel<phi::GPUContext, int>,
                        ops::Flatten2Kernel<phi::GPUContext, int8_t>,
                        ops::Flatten2Kernel<phi::GPUContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(flatten2_grad,
                        ops::Flatten2GradKernel<phi::GPUContext, float>,
                        ops::Flatten2GradKernel<phi::GPUContext, double>,
                        ops::Flatten2GradKernel<phi::GPUContext, uint8_t>,
                        ops::Flatten2GradKernel<phi::GPUContext, int>,
                        ops::Flatten2GradKernel<phi::GPUContext, int8_t>,
                        ops::Flatten2GradKernel<phi::GPUContext, int64_t>);
