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
// #include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;

template <typename T>
class SquaredL2NormMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &dev_ctx = context.template device_context<MLUDeviceContext>();
    auto *x = context.Input<phi::DenseTensor>("X");
    auto *out = context.Output<phi::DenseTensor>("Out");

    auto place = context.GetPlace();

    out->mutable_data<T>(place);

    MLUCnnlTensorDesc input_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);

    // L2Loss
    MLUCnnl::L2Loss(context, input_desc.get(), GetBasePtr(x), GetBasePtr(out));

    // do mul
    phi::DenseTensor scale_tensor =
        context.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    phi::DenseTensor bias_tensor =
        context.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc bias_desc(bias_tensor);
    FillMLUTensorWithHostValue(context, static_cast<T>(2.0f), &scale_tensor);
    FillMLUTensorWithHostValue(context, static_cast<T>(0.0f), &bias_tensor);

    MLUCnnl::Scale(context,
                   0,
                   out_desc.get(),
                   GetBasePtr(out),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_desc.get(),
                   GetBasePtr(out));
  }
};

template <typename T>
class SquaredL2NormGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto &dev_ctx = context.template device_context<MLUDeviceContext>();
    auto *x = context.Input<phi::DenseTensor>("X");
    auto *x_grad =
        context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto *out_grad =
        context.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    PADDLE_ENFORCE_EQ(
        out_grad->numel(),
        1,
        platform::errors::InvalidArgument(
            "Input(GRAD@Out) of SquaredL2NormGradOP should be a scalar."));

    auto place = context.GetPlace();

    // broadcast out_grad
    Tensor broadcasted_out_grad;
    broadcasted_out_grad.mutable_data<T>(x_grad->dims(), place);
    MLUCnnlTensorDesc broadcasted_out_grad_desc(broadcasted_out_grad);
    MLUCnnlTensorDesc out_grad_desc(*out_grad);
    MLUCnnl::BroadcastTo(context,
                         out_grad_desc.get(),
                         GetBasePtr(out_grad),
                         broadcasted_out_grad_desc.get(),
                         GetBasePtr(&broadcasted_out_grad));

    // mul x
    Tensor tmp_x_grad;
    tmp_x_grad.mutable_data<T>(x_grad->dims(), place);
    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc tmp_x_grad_desc(tmp_x_grad);
    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType(x->dtype()), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(context,
                      mul_op_desc.get(),
                      x_desc.get(),
                      GetBasePtr(x),
                      broadcasted_out_grad_desc.get(),
                      GetBasePtr(&broadcasted_out_grad),
                      tmp_x_grad_desc.get(),
                      GetBasePtr(&tmp_x_grad),
                      ToCnnlDataType(x->dtype()));

    // mul
    phi::DenseTensor scale_tensor =
        context.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    phi::DenseTensor bias_tensor =
        context.AllocateTmpTensor<T, MLUDeviceContext>({1}, dev_ctx);
    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnlTensorDesc bias_desc(bias_tensor);
    FillMLUTensorWithHostValue(context, static_cast<T>(2.0f), &scale_tensor);
    FillMLUTensorWithHostValue(context, static_cast<T>(0.0f), &bias_tensor);

    x_grad->mutable_data<T>(place);
    MLUCnnlTensorDesc x_grad_desc(*x_grad);
    MLUCnnl::Scale(context,
                   0,
                   tmp_x_grad_desc.get(),
                   GetBasePtr(&tmp_x_grad),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   x_grad_desc.get(),
                   GetBasePtr(x_grad));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(squared_l2_norm,
                       ops::SquaredL2NormMLUKernel<float>,
                       ops::SquaredL2NormMLUKernel<plat::float16>);
REGISTER_OP_MLU_KERNEL(squared_l2_norm_grad,
                       ops::SquaredL2NormGradMLUKernel<float>,
                       ops::SquaredL2NormGradMLUKernel<plat::float16>);
