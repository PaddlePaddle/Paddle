/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/sampling_id_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SamplingIdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {}
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    sampling_id,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SamplingIdKernel<paddle::platform::CUDADeviceContext, int64_t>);
