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
namespace p = paddle::platform;

template <typename DeviceContext, typename T>
class AccMergeXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<phi::XPUContext>();
    xpu::Context* xpu_ctx = dev_ctx.x_context();

    phi::DenseTensor &step_t = *ctx.Output<phi::DenseTensor>("Step");
    auto *step = step_t.data<int64_t>();
    if (step[1] <= 0) return;

    const phi::DenseTensor &total_t = *ctx.Input<phi::DenseTensor>("Total");
    bool is_cpu_place = p::is_cpu_place(total_t.place());

    using Type1 = float;
    using Type2 = double;

    const phi::DenseTensor &acc_t = *ctx.Input<phi::DenseTensor>("Acc");
    auto *acc = acc_t.data<Type1>();

    phi::DenseTensor &out_t = *ctx.Output<phi::DenseTensor>("Out");
    out_t.Resize({2});
    auto *out = out_t.mutable_data<Type2>(acc_t.place());
    int r = 0;
    if (step[0] == 0) {
      if (is_cpu_place) {
        r = xpu::acc_merge_cpu<Type1, Type2, false>(xpu_ctx, acc, *total_t.data<int64_t>(), out);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "acc_merge_cpu");
      } else {
        r = xpu::acc_merge_xpu<Type1, Type2, false>(xpu_ctx, acc, total_t.data<float>(), out);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "acc_merge_xpu");
      }
    } else {
      if (is_cpu_place) {
        r = xpu::acc_merge_cpu<Type1, Type2, true>(xpu_ctx, acc, *total_t.data<int64_t>(), out);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "acc_merge_cpu");
      } else {
        r = xpu::acc_merge_xpu<Type1, Type2, true>(xpu_ctx, acc, total_t.data<float>(), out);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "acc_merge_xpu");
      }
    }
    step[0] = (step[0] + 1) % step[1];
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
