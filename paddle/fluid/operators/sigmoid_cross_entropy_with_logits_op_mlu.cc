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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
const int kIgnoreIndex = -100;

void CheckAttrs(const framework::ExecutionContext& ctx) {
  // cnnl not support normalize and ignore_index
  bool normalize = ctx.Attr<bool>("normalize");
  int ignore_index = ctx.Attr<int>("ignore_index");
  PADDLE_ENFORCE_EQ(normalize,
                    false,
                    platform::errors::InvalidArgument(
                        "attr normalize must be false, but got true"));
  PADDLE_ENFORCE_EQ(ignore_index,
                    kIgnoreIndex,
                    platform::errors::InvalidArgument(
                        "attr ignore_index must be default %d, but got %d",
                        kIgnoreIndex,
                        ignore_index));
}

template <typename T>
class SigmoidCrossEntropyWithLogitsMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    CheckAttrs(ctx);

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* label = ctx.Input<phi::DenseTensor>("Label");

    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto place = ctx.GetPlace();

    out->mutable_data<T>(place);

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc label_desc(*label);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::BceWithLogits(ctx,
                           CNNL_BCE_WITH_LOGITS_NONE,
                           x_desc.get(),
                           GetBasePtr(x),
                           label_desc.get(),
                           GetBasePtr(label),
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           out_desc.get(),
                           GetBasePtr(out));
  }
};

template <typename T>
class SigmoidCrossEntropyWithLogitsMLUGradKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    CheckAttrs(ctx);

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* label = ctx.Input<phi::DenseTensor>("Label");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto place = ctx.GetPlace();

    dx->mutable_data<T>(place);

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc label_desc(*label);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnl::BceWithLogitsBackward(ctx,
                                   CNNL_BCE_WITH_LOGITS_NONE,
                                   dout_desc.get(),
                                   GetBasePtr(dout),
                                   x_desc.get(),
                                   GetBasePtr(x),
                                   label_desc.get(),
                                   GetBasePtr(label),
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   x_desc.get(),
                                   GetBasePtr(dx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_MLU_KERNEL(
    sigmoid_cross_entropy_with_logits,
    ops::SigmoidCrossEntropyWithLogitsMLUKernel<float>,
    ops::SigmoidCrossEntropyWithLogitsMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    ops::SigmoidCrossEntropyWithLogitsMLUGradKernel<float>,
    ops::SigmoidCrossEntropyWithLogitsMLUGradKernel<plat::float16>);
