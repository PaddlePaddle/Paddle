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

#include "paddle/fluid/operators/detection/iou_similarity_op.h"

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class XPUIOUSimilarityKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const phi::DenseTensor* in_x = ctx.Input<phi::DenseTensor>("X");
    const phi::DenseTensor* in_y = ctx.Input<phi::DenseTensor>("Y");
    bool normalized = ctx.Attr<bool>("box_normalized");
    phi::DenseTensor* out = ctx.Output<phi::DenseTensor>("Out");

    int x_n = in_x->dims()[0];
    int y_n = in_y->dims()[0];
    T eps = static_cast<T>(1e-10);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::iou_similarity(dev_ctx.x_context(),
                                in_x->data<T>(),
                                in_y->data<T>(),
                                out->mutable_data<T>(ctx.GetPlace()),
                                x_n,
                                y_n,
                                eps,
                                normalized);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        phi::errors::External(
            "XPU iou_similarity kernel return wrong value[%d %s].",
            r,
            XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using XPU = paddle::platform::XPUDeviceContext;

PD_REGISTER_STRUCT_KERNEL(
    iou_similarity, XPU, ALL_LAYOUT, ops::XPUIOUSimilarityKernel, float) {}

#endif
