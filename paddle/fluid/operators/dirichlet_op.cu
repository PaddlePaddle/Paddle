// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/dirichlet_op.h"

namespace paddle {
namespace operators {
template <typename T>
struct DirichletSampler<platform::CUDADeviceContext, T> {
  void operator()(const framework::ExecutionContext& ctx, const Tensor* alpha,
                  Tensor* out) {}
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    dirichlet, ops::DirichletKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DirichletKernel<paddle::platform::CUDADeviceContext, double>);
