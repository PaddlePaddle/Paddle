/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include <memory>
#include <string>

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <cnnlActivationMode_t act_mode, typename T>
class ActivationMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    output->mutable_data<T>(ctx.GetPlace());

    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnlTensorDesc input_desc(*input);
    MLUCnnlTensorDesc output_desc(*output);

    MLUCnnl::Active(ctx, act_desc.get(), input_desc.get(), GetBasePtr(input),
                    output_desc.get(), GetBasePtr(output));
  }
};

// For gelu, leaky_relu
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV1 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx, act_desc.get(), nullptr, nullptr, nullptr, nullptr,
                        dout_desc.get(), GetBasePtr(dout), x_desc.get(),
                        GetBasePtr(x), dx_desc.get(), GetBasePtr(dx));
  }
};

// For tanh, sigmoid
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV2 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx, act_desc.get(), nullptr, nullptr, out_desc.get(),
                        GetBasePtr(out), dout_desc.get(), GetBasePtr(dout),
                        nullptr, nullptr, dx_desc.get(), GetBasePtr(dx));
  }
};

// For relu, relu6
template <cnnlActivationMode_t act_mode, typename T>
class ActivationGradMLUKernelV3 : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 1.0f;

    dx->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc dout_desc(*dout);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlActivationDesc act_desc(act_mode, alpha);
    MLUCnnl::ActiveGrad(ctx, act_desc.get(), nullptr, nullptr, nullptr, nullptr,
                        dout_desc.get(), GetBasePtr(dout), out_desc.get(),
                        GetBasePtr(out), dx_desc.get(), GetBasePtr(dx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

// relu
REGISTER_OP_MLU_KERNEL(
    relu, ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu_grad, ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU, float>,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU,
                                   paddle::platform::float16>);

// relu6
REGISTER_OP_MLU_KERNEL(
    relu6, ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU6, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_RELU6, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    relu6_grad, ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU6, float>,
    ops::ActivationGradMLUKernelV3<CNNL_ACTIVATION_RELU6,
                                   paddle::platform::float16>);

// sigmoid
REGISTER_OP_MLU_KERNEL(sigmoid,
                       ops::ActivationMLUKernel<CNNL_ACTIVATION_SIGMOID, float>,
                       ops::ActivationMLUKernel<CNNL_ACTIVATION_SIGMOID,
                                                paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    sigmoid_grad,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_SIGMOID, float>,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_SIGMOID,
                                   paddle::platform::float16>);

// tanh
REGISTER_OP_MLU_KERNEL(
    tanh, ops::ActivationMLUKernel<CNNL_ACTIVATION_TANH, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_TANH, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    tanh_grad, ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_TANH, float>,
    ops::ActivationGradMLUKernelV2<CNNL_ACTIVATION_TANH,
                                   paddle::platform::float16>);

// gelu
REGISTER_OP_MLU_KERNEL(
    gelu, ops::ActivationMLUKernel<CNNL_ACTIVATION_GELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_GELU, paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    gelu_grad, ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_GELU, float>,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_GELU,
                                   paddle::platform::float16>);

// leaky_relu
REGISTER_OP_MLU_KERNEL(
    leaky_relu, ops::ActivationMLUKernel<CNNL_ACTIVATION_LEAKYRELU, float>,
    ops::ActivationMLUKernel<CNNL_ACTIVATION_LEAKYRELU,
                             paddle::platform::float16>);
REGISTER_OP_MLU_KERNEL(
    leaky_relu_grad,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_LEAKYRELU, float>,
    ops::ActivationGradMLUKernelV1<CNNL_ACTIVATION_LEAKYRELU,
                                   paddle::platform::float16>);
