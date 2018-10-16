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

#include "paddle/fluid/operators/roi_align_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaxinumNumBlocks = 4096;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    roi_align,
    ops::GPUROIAlignOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIAlignOpKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    roi_align_grad,
    ops::GPUROIAlignGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GPUROIAlignGradOpKernel<paddle::platform::CUDADeviceContext, double>);
