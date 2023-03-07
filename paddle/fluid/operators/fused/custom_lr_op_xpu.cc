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
class CustomLrXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // PADDLE_THROW(platform::errors::Unimplemented(
    // "The custom_lr operator does not support XPU yet."));
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    const phi::DenseTensor* x = ctx.Input<phi::DenseTensor>("X");

    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");

    float* out_ptr = out->mutable_data<float>(ctx.GetPlace());
    bool base_lr = ctx.Attr<float>("base_lr");
    bool max_step = ctx.Attr<int64_t>("max_step");

    int r = xpu::constant<float>(xpu_ctx, out_ptr, 1, 0.1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");

    (void)dev_ctx;
    (void)x;
    (void)out_ptr;
    (void)base_lr;
    (void)max_step;

    VLOG(5) << "base_lr = " << base_lr << " , max_step = " << max_step;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(custom_lr,
                       ops::CustomLrXPUKernel<phi::XPUContext, int64_t>);
