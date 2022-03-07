/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace operators {

template <typename T>
class MaskedSelectXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto input = context.Input<framework::Tensor>("X");
    auto mask = context.Input<framework::Tensor>("Mask");
    auto out = context.Output<framework::Tensor>("Y");
    auto* mask_data = mask->data<bool>();
    auto* input_data = input->data<T>();
    auto input_dim = input->dims();
    auto mask_dim = mask->dims();
    PADDLE_ENFORCE_EQ(
        input_dim, mask_dim,
        platform::errors::InvalidArgument(
            "The dim size of input and mask in OP(masked_selected) "
            "must be equal, but got input dim:(%ld), mask dim: "
            "(%ld). Please check input "
            "value.",
            input_dim, mask_dim));
    auto& dev_ctx =
        context.template device_context<paddle::platform::XPUDeviceContext>();
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* out_size = RAII_GUARD.alloc_l3_or_gm<int32_t>(1);
    int out_size_cpu;

    PADDLE_ENFORCE_XDNN_SUCCESS(
        xpu::nonzero_count(dev_ctx.x_context(), mask_data, out_size,
                           mask->numel()),
        "nonzero_count ");
    memory::Copy(platform::CPUPlace(), static_cast<void*>(&out_size_cpu),
                 mask->place(), static_cast<void*>(out_size), sizeof(int32_t));

    framework::DDim out_dim{out_size_cpu};
    out->Resize(out_dim);
    auto out_data = out->mutable_data<T>(context.GetPlace());

    auto input_shape = phi::vectorize<int>(input_dim);
    auto mask_shape = phi::vectorize<int>(mask_dim);

    PADDLE_ENFORCE_XDNN_SUCCESS(
        xpu::masked_select(dev_ctx.x_context(), input_data, mask_data, out_data,
                           input_shape, mask_shape, out_size_cpu),
        "masked_select");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(masked_select, ops::MaskedSelectXPUKernel<float>,
                       ops::MaskedSelectXPUKernel<int>,
                       ops::MaskedSelectXPUKernel<int64_t>);
#endif
