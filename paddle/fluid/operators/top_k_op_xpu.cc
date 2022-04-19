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

#ifdef PADDLE_WITH_XPU

#include <memory>

#include "paddle/fluid/operators/top_k_op.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
class TopkXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Get the top k elements of each row of input tensor
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");

    size_t k = static_cast<int>(ctx.Attr<int>("k"));
    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      k = k_t->data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[output_dims.size() - 1] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());
    Tensor indices_32_data_tensor;
    int32_t* indices_int_data = indices_32_data_tensor.mutable_data<int32_t>(
        ctx.GetPlace(), indices->numel());
    // reshape input to a flattern matrix(like flat_inner_dims)
    framework::DDim inputdims = input->dims();
    const size_t row =
        phi::product(phi::slice_ddim(inputdims, 0, inputdims.size() - 1));
    const size_t col = inputdims[inputdims.size() - 1];
    auto& dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();

    int ret = xpu::sorted_topk<T>(dev_ctx.x_context(), input->data<T>(),
                                  output_data, indices_int_data, row, col, k);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU API return wrong value[%d] in call kernel name "
                          "[%s], please check "
                          "where Baidu Kunlun Card is properly installed.",
                          ret, "sorted_topk"));
    ret = xpu::cast_v2<int32_t, int64_t>(dev_ctx.x_context(),
                                         (const int32_t*)indices_int_data,
                                         indices_data, indices->numel());
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      platform::errors::External(
                          "XPU API return wrong value[%d] in call kernel name "
                          "[%s], please check "
                          "where Baidu Kunlun Card is properly installed.",
                          ret, "cast_v2"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(top_k, ops::TopkXPUKernel<float>);
#endif
