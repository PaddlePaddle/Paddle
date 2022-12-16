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

#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class TransposeMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    out->mutable_data<T>(ctx.device_context().GetPlace());

    TransposeFromMLUTensor<T>(
        ctx, axis, x, out, false /*need_reshape_or_alloc*/);
  }
};

template <typename T>
class TransposeGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }
    x_grad->mutable_data<T>(ctx.GetPlace());

    TransposeFromMLUTensor<T>(
        ctx, reversed_axis, out_grad, x_grad, false /*need_reshape_or_alloc*/);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_MLU_KERNEL(transpose2,
                       ops::TransposeMLUKernel<float>,
                       ops::TransposeMLUKernel<paddle::platform::float16>,
                       ops::TransposeMLUKernel<int>,
                       ops::TransposeMLUKernel<int16_t>,
                       ops::TransposeMLUKernel<uint8_t>,
                       ops::TransposeMLUKernel<int8_t>,
                       ops::TransposeMLUKernel<bool>);

REGISTER_OP_MLU_KERNEL(transpose2_grad,
                       ops::TransposeGradMLUKernel<float>,
                       ops::TransposeGradMLUKernel<paddle::platform::float16>,
                       ops::TransposeGradMLUKernel<int>,
                       ops::TransposeGradMLUKernel<int16_t>,
                       ops::TransposeGradMLUKernel<uint8_t>,
                       ops::TransposeGradMLUKernel<int8_t>,
                       ops::TransposeGradMLUKernel<bool>);
