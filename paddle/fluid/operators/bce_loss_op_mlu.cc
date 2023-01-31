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

template <typename T>
class BCELossMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* labels = ctx.Input<phi::DenseTensor>("Label");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc label_desc(*labels);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::BceLoss(ctx,
                     CNNL_BCE_LOSS_NONE,
                     x_desc.get(),
                     GetBasePtr(x),
                     label_desc.get(),
                     GetBasePtr(labels),
                     nullptr,
                     nullptr,
                     out_desc.get(),
                     GetBasePtr(out));
  }
};

template <typename T>
class BCELossGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* labels = ctx.Input<phi::DenseTensor>("Label");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc label_desc(*labels);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnl::BceLossBackward(ctx,
                             CNNL_BCE_LOSS_NONE,
                             dout_desc.get(),
                             GetBasePtr(dout),
                             x_desc.get(),
                             GetBasePtr(x),
                             label_desc.get(),
                             GetBasePtr(labels),
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

REGISTER_OP_MLU_KERNEL(bce_loss,
                       ops::BCELossMLUKernel<float>,
                       ops::BCELossMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(bce_loss_grad,
                       ops::BCELossGradMLUKernel<float>,
                       ops::BCELossGradMLUKernel<plat::float16>);
