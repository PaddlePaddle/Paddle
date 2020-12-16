/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

#define DECLARE_OP(op_name)                                             \
  class op_name##Op : public framework::OperatorWithKernel {            \
   public:                                                              \
    using framework::OperatorWithKernel::OperatorWithKernel;            \
    void InferShape(framework::InferShapeContext* ctx) const override { \
      OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", #op_name);       \
      OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", #op_name); \
      auto x_dims = ctx->GetInputDim("X");                              \
      ctx->SetOutputDim("Out", x_dims);                                 \
      ctx->ShareLoD("X", "Out");                                        \
    }                                                                   \
  }

#define DECLARE_OP_MAKER(op_name)                                          \
  class op_name##OpMaker : public framework::OpProtoAndCheckerMaker {      \
   public:                                                                 \
    void Make() override {                                                 \
      AddInput("X", "(Tensor), The input tensor of " #op_name " op.");     \
      AddOutput("Out", "(Tensor), The output tensor of " #op_name " op."); \
      AddComment(string::Sprintf(R"DOC( \
  %s Operator. \
  This operator is used to get a new tensor containing %s values \
  from a tensor with complex data type. \
  )DOC",                                                                   \
                                 #op_name, #op_name));                     \
    }                                                                      \
  }

#define DECLARE_GRAD_OP(op_name)                                            \
  class op_name##GradOp : public framework::OperatorWithKernel {            \
   public:                                                                  \
    using framework::OperatorWithKernel::OperatorWithKernel;                \
    void InferShape(framework::InferShapeContext* ctx) const override {     \
      OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", #op_name "Grad");    \
      OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input", \
                     "Out@Grad", #op_name "Grad");                          \
      auto x_dims = ctx->GetInputDim("X");                                  \
      auto x_grad_name = framework::GradVarName("X");                       \
      if (ctx->HasOutput(x_grad_name)) {                                    \
        ctx->SetOutputDim(x_grad_name, x_dims);                             \
      }                                                                     \
    }                                                                       \
                                                                            \
   protected:                                                               \
    framework::OpKernelType GetExpectedKernelType(                          \
        const framework::ExecutionContext& ctx) const override {            \
      auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");   \
      return framework::OpKernelType(data_type, ctx.GetPlace());            \
    }                                                                       \
  }

#define DECLARE_GRAD_OP_MAKER(op_name, op_type)                              \
  template <typename T>                                                      \
  class op_name##GradOpMaker : public framework::SingleGradOpMaker<T> {      \
   public:                                                                   \
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;                \
    void Apply(GradOpPtr<T> grad_op) const override {                        \
      grad_op->SetType(#op_type "_grad");                                    \
      grad_op->SetInput("X", this->Input("X"));                              \
      grad_op->SetInput(framework::GradVarName("Out"),                       \
                        this->OutputGrad("Out"));                            \
      grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X")); \
    }                                                                        \
  }

#define DECLARE_OP_KERNEL(op_name)                                            \
  template <typename DeviceContext, typename T>                               \
  class op_name##Kernel : public framework::OpKernel<T> {                     \
    using op_name##T = math::select_t<                                        \
        math::cond<std::is_same<T, platform::complex64>::value, float>,       \
        math::cond<std::is_same<T, platform::complex128>::value, double>, T>; \
                                                                              \
   public:                                                                    \
    void Compute(const framework::ExecutionContext& ctx) const {              \
      const framework::Tensor* x = ctx.Input<framework::Tensor>("X");         \
      framework::Tensor* out = ctx.Output<framework::Tensor>("Out");          \
      auto numel = x->numel();                                                \
      auto* x_data = x->data<T>();                                            \
      auto* out_data = out->mutable_data<op_name##T>(                         \
          ctx.GetPlace(), static_cast<size_t>(numel * sizeof(op_name##T)));   \
      auto& dev_ctx = ctx.template device_context<DeviceContext>();           \
      platform::ForRange<DeviceContext> for_range(dev_ctx, numel);            \
      math::op_name##Functor<T> functor(x_data, out_data, numel);             \
      for_range(functor);                                                     \
    }                                                                         \
  }

#define DECLARE_GRAD_OP_KERNEL(op_name)                                       \
  template <typename DeviceContext, typename T>                               \
  class op_name##GradKernel : public framework::OpKernel<T> {                 \
    using op_name##T = math::select_t<                                        \
        math::cond<std::is_same<T, platform::complex64>::value, float>,       \
        math::cond<std::is_same<T, platform::complex128>::value, double>, T>; \
                                                                              \
   public:                                                                    \
    void Compute(const framework::ExecutionContext& ctx) const {              \
      const framework::Tensor* d_out =                                        \
          ctx.Input<framework::Tensor>(framework::GradVarName("Out"));        \
      framework::Tensor* d_x =                                                \
          ctx.Output<framework::Tensor>(framework::GradVarName("X"));         \
      auto numel = d_out->numel();                                            \
      auto* dout_data = d_out->data<op_name##T>();                            \
      auto* dx_data = d_x->mutable_data<T>(                                   \
          ctx.GetPlace(), static_cast<size_t>(numel * sizeof(T)));            \
      auto& dev_ctx = ctx.template device_context<DeviceContext>();           \
      platform::ForRange<DeviceContext> for_range(dev_ctx, numel);            \
      math::op_name##ToComplexFunctor<T> functor(dout_data, dx_data, numel);  \
      for_range(functor);                                                     \
    }                                                                         \
  }
}  // namespace operators
}  // namespace paddle

#define REGISTER_OP(op_name, op_type)                                         \
  REGISTER_OPERATOR(                                                          \
      op_type, ::paddle::operators::op_name##Op,                              \
      ::paddle::operators::op_name##OpMaker,                                  \
      ::paddle::operators::op_name##GradOpMaker<::paddle::framework::OpDesc>, \
      ::paddle::operators::op_name##GradOpMaker<::paddle::imperative::OpBase>)

#define REGISTER_GRAD_OP(op_name, op_type) \
  REGISTER_OPERATOR(op_type, ::paddle::operators::op_name##Op)

#define REGISTER_OP_CPU_COMPLEX_KERNEL(op_name, op_type)                    \
  REGISTER_OP_CPU_KERNEL(op_type, ::paddle::operators::op_name##Kernel<     \
                                      ::paddle::platform::CPUDeviceContext, \
                                      ::paddle::platform::complex64>,       \
                         ::paddle::operators::op_name##Kernel<              \
                             ::paddle::platform::CPUDeviceContext,          \
                             ::paddle::platform::complex128>)

#define REGISTER_OP_CUDA_COMPLEX_KERNEL(op_name, op_type)                     \
  REGISTER_OP_CUDA_KERNEL(op_type, ::paddle::operators::op_name##Kernel<      \
                                       ::paddle::platform::CUDADeviceContext, \
                                       ::paddle::platform::complex64>,        \
                          ::paddle::operators::op_name##Kernel<               \
                              ::paddle::platform::CUDADeviceContext,          \
                              ::paddle::platform::complex128>)
