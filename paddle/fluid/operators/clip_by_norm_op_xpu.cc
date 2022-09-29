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
#include <vector>

#include "paddle/fluid/operators/clip_by_norm_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPUClipByNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto max_norm = context.Attr<T>("max_norm");
    auto in_var = context.InputVar("X");

    phi::DenseTensor* output = nullptr;
    const phi::DenseTensor* input = nullptr;
    if (in_var->IsType<framework::LoDTensor>()) {
      input = context.Input<phi::DenseTensor>("X");

      output = context.Output<phi::DenseTensor>("Out");
      output->mutable_data<T>(context.GetPlace());
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Invalid input variable type, only support LodTensor"
          "type, but got type is %s.",
          framework::ToTypeName(in_var->Type())));
    }

    PADDLE_ENFORCE_NOT_NULL(input,
                            platform::errors::InvalidArgument(
                                "Input(X) of ClipByNormOp should not be null. "
                                "Please check if it is created correctly."));
    auto& dev_ctx = context.template device_context<DeviceContext>();
    const auto& x_dims = input->dims();
    std::vector<int> xshape(x_dims.size());
    std::vector<int> rdims(x_dims.size());
    for (int i = 0; i < x_dims.size(); i++) {
      xshape[i] = x_dims[i];
      rdims[i] = i;
    }
    int r = xpu::clip_by_norm<T>(dev_ctx.x_context(),
                                 input->data<T>(),
                                 output->data<T>(),
                                 max_norm,
                                 xshape,
                                 rdims);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        platform::errors::External("XPU API(clip_by_norm) return "
                                   "wrong value[%d], please check whether "
                                   "Baidu Kunlun Card is properly installed.",
                                   r));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    clip_by_norm,
    ops::XPUClipByNormKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
