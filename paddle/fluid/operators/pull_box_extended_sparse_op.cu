//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/pull_box_extended_sparse_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class PullBoxExtendedSparseCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxExtendedSparseFunctor<T>(ctx);
  }
};

template <typename T, typename DeviceContext>
class PushBoxExtendedSparseCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxExtendedSparseFunctor<T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

PD_REGISTER_STRUCT_KERNEL(pull_box_extended_sparse,
                          GPU,
                          ALL_LAYOUT,
                          ops::PullBoxExtendedSparseCUDAKernel,
                          float,
                          double) {}
PD_REGISTER_STRUCT_KERNEL(push_box_extended_sparse,
                          GPU,
                          ALL_LAYOUT,
                          ops::PushBoxExtendedSparseCUDAKernel,
                          float,
                          double) {}
