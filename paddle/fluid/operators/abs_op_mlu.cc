/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class AbsMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<phi::DenseTensor>("X");
    auto* output = ctx.Output<phi::DenseTensor>("Out");

    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Abs(ctx,
                 input_desc.get(),
                 GetBasePtr(input),
                 output_desc.get(),
                 GetBasePtr(output));
  }
};

template <typename T>
class AbsGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc input_desc(*x);
    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

    Tensor sign_x;
    sign_x.mutable_data<T>(x->dims(), ctx.GetPlace());

    MLUCnnl::Sign(ctx,
                  input_desc.get(),
                  GetBasePtr(x),
                  input_desc.get(),
                  GetBasePtr(&sign_x));
    MLUCnnl::OpTensor(ctx,
                      mul_op_desc.get(),
                      input_desc.get(),
                      GetBasePtr(&sign_x),
                      input_desc.get(),
                      GetBasePtr(dout),
                      input_desc.get(),
                      GetBasePtr(dx),
                      ToCnnlDataType<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(abs,
                       ops::AbsMLUKernel<float>,
                       ops::AbsMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(abs_grad,
                       ops::AbsGradMLUKernel<float>,
                       ops::AbsGradMLUKernel<plat::float16>);
