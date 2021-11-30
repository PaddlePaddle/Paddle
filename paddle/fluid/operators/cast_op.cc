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

#include "paddle/fluid/operators/cast_op.h"
#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

class CastOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of cast op");
    AddOutput("Out", "The output tensor of cast op");
    AddAttr<int>("out_dtype", "output data type");
    AddAttr<int>("in_dtype", "input data type");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddComment(R"DOC(
Cast Operator.

This Operator casts the input tensor to another data type and
returns the Output Tensor. It's meaningless if the output dtype equals
the input dtype, but it's fine if you do so.

)DOC");
  }
};

template <typename T>
class CastOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad) const override {
    grad->SetType("cast");
    grad->SetInput("X", this->OutputGrad("Out"));
    grad->SetOutput("Out", this->InputGrad("X"));
    grad->SetAttr("out_dtype", this->GetAttr("in_dtype"));
    grad->SetAttr("in_dtype", this->GetAttr("out_dtype"));
    grad->SetAttr("use_mkldnn", this->GetAttr("use_mkldnn"));
  }
};

class CastOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "cast");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "cast");
    context->SetOutputDim("Out", context->GetInputDim("X"));
    context->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // CastOp kernel's device type is decided by input tensor place
    auto *tensor = ctx.Input<framework::LoDTensor>("X");
    PADDLE_ENFORCE_EQ(tensor->IsInitialized(), true,
                      platform::errors::PreconditionNotMet(
                          "The tensor of Input(X) is not initialized."));
    auto &tensor_place = tensor->place();
    // NOTE: cuda pinned tensor need to copy its data to target place
    if (platform::is_cuda_pinned_place(tensor_place)) {
      return framework::OpKernelType(tensor->type(), ctx.device_context());
    }

#ifdef PADDLE_WITH_MKLDNN
    int in_dtype = ctx.Attr<int>("in_dtype");
    int out_dtype = ctx.Attr<int>("out_dtype");

    auto MKLDNNSupportsCast = [&]() -> bool {
      int dtype_fp32 = static_cast<int>(framework::proto::VarType::FP32);
      int dtype_bf16 = static_cast<int>(framework::proto::VarType::BF16);

      if ((in_dtype != dtype_fp32 && in_dtype != dtype_bf16) ||
          (out_dtype != dtype_fp32 && out_dtype != dtype_bf16))
        return false;

      return true;
    };

    if (this->CanMKLDNNBeUsed(ctx, tensor->type()) && MKLDNNSupportsCast()) {
      return framework::OpKernelType(tensor->type(), ctx.GetPlace(),
                                     framework::DataLayout::kMKLDNN,
                                     framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(tensor->type(), tensor_place);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;
#define REGISTER_CAST_CPU_BASE(op_name, ...)                                  \
  REGISTER_OPERATOR(op_name, ops::CastOp,                                     \
                    ops::CastOpGradMaker<paddle::framework::OpDesc>,          \
                    ops::CastOpGradMaker<paddle::imperative::OpBase>,         \
                    ops::CastOpProtoMaker);                                   \
  REGISTER_OP_CPU_KERNEL(                                                     \
      op_name, ops::CastOpKernel<CPU, float>, ops::CastOpKernel<CPU, double>, \
      ops::CastOpKernel<CPU, int>, ops::CastOpKernel<CPU, int64_t>,           \
      ops::CastOpKernel<CPU, int>, ops::CastOpKernel<CPU, int16_t>,           \
      ops::CastOpKernel<CPU, bool>, ops::CastOpKernel<CPU, uint8_t>,          \
      ops::CastOpKernel<CPU, paddle::platform::float16>,                      \
      ops::CastOpKernel<CPU, paddle::platform::bfloat16>,                     \
      ops::CastOpKernel<CPU, paddle::platform::complex<float>>,               \
      ops::CastOpKernel<CPU, paddle::platform::complex<double>>);

REGISTER_CAST_CPU_BASE(cast)
// [ why register transfer_dtype_op alias with cast_op? ]
// In case of InterpreterCore, if we reuse cast_op, we cannot distinguish
// which cast_op is inserted by new executor when we do profiling.
REGISTER_CAST_CPU_BASE(transfer_dtype)
