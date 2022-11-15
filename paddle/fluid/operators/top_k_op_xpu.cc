/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>

#include "paddle/fluid/operators/top_k_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
template <typename T>
class TopkXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    const auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");
    auto* indices = ctx.Output<phi::DenseTensor>("Indices");

    // get k from attr
    int k = static_cast<int>(ctx.Attr<int>("k"));

    // get k from input tensor
    auto* k_t = ctx.Input<phi::DenseTensor>("K");
    if (k_t) {
      memory::Copy(platform::CPUPlace(),
                   static_cast<void*>(&k),
                   ctx.GetPlace(),
                   static_cast<const void*>(k_t->data<int>()),
                   sizeof(int));
      framework::DDim output_dims = output->dims();
      output_dims[output_dims.size() - 1] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    // allocate temp memory for int32 index
    xpu::ctx_guard RAII_GUARD(dev_ctx.x_context());
    int* indices_int_data = RAII_GUARD.alloc_l3_or_gm<int>(indices->numel());
    PADDLE_ENFORCE_XDNN_NOT_NULL(indices_int_data);

    // reshape input to a flattern matrix(like flat_inner_dims)
    framework::DDim inputdims = input->dims();
    const size_t row =
        phi::product(phi::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const size_t col = inputdims[inputdims.size() - 1];

    // int sorted_topk(Context* ctx, const T* x, T* y, int* index, int m, int n,
    // int k, bool largest = true);
    int r = xpu::sorted_topk(dev_ctx.x_context(),
                             reinterpret_cast<const XPUType*>(input->data<T>()),
                             reinterpret_cast<XPUType*>(output_data),
                             indices_int_data,
                             row,
                             col,
                             k);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "sorted_topk");

    // cast to int64 as final result
    r = xpu::cast_v2<int32_t, int64_t>(dev_ctx.x_context(),
                                       (const int32_t*)indices_int_data,
                                       indices_data,
                                       indices->numel());
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast_v2");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(top_k,
                       ops::TopkXPUKernel<float>,
                       ops::TopkXPUKernel<paddle::platform::float16>);
#endif
