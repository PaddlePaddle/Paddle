/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/scope_guard.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/xpu/xpu_api_wrapper.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename DeviceContext, typename T>
class AccMergeXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "The acc_merge operator does not support XPU yet."));
    // auto& dev_ctx = ctx.template device_context<phi::XPUContext>();

    // const phi::DenseTensor* acc = ctx.Input<phi::DenseTensor>("Acc");
    // const phi::DenseTensor* total = ctx.Input<phi::DenseTensor>("Total");

    // phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");
    // phi::DenseTensor* step = ctx.Output<phi::DenseTensor>("Step");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    acc_merge,
    ops::AccMergeXPUKernel<phi::XPUContext, float>,
    ops::AccMergeXPUKernel<phi::XPUContext,
                                    paddle::platform::float16>);
