/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/flatten_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FlattenOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input (X) of Flatten op should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output (Output) of Flatten op should not be null.");
    const auto &axis = ctx->Attrs().Get<int>("axis");
    const auto &in_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(axis, 0,
                      "The axis should be greater than or equal to 0.");
    PADDLE_ENFORCE_LE(
        axis, in_dims.size(),
        "The axis should be less than or equal to input tensor's rank.");

    const auto &out_dims = GetOutputShape(axis, in_dims);
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    if (in_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

  static std::vector<int32_t> GetOutputShape(const int axis,
                                             const framework::DDim &in_dims) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < in_dims.size(); ++i) {
      if (i < axis) {
        outer *= in_dims[i];
      } else {
        inner *= in_dims[i];
      }
    }
    std::vector<int32_t> out_shape(2);
    out_shape[0] = outer;
    out_shape[1] = inner;
    return out_shape;
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FlattenOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) A tensor of rank >= axis.");
    AddOutput("Out",
              "A 2D tensor is reshaped input tensor. The input dimensions"
              "up to axis are flattened to the outer dimension of the output"
              "and the remaining input dimensions are flattened into the inner"
              "dimension of the output.");
    AddAttr<int>("axis",
                 "(int)"
                 "Indicate up to which input dimensions (exclusive) should be"
                 "flattened to the outer dimension of the output. The value"
                 "for axis must be in the range [0, R], where R is the rank of"
                 "the input tensor. When axis = 0, the shape of the output"
                 "tensor is (1, (d_0 X d_1 ... d_n), where the shape of the"
                 "input tensor is (d_0, d_1, ... d_n).")
        .SetDefault(1);
    AddComment(R"DOC(
Flatten Operator

Flattens the input tensor into a 2D matrix.

Examples:
Case 1:
  Given
    X.shape = (3, 100, 100, 4)
  and
    axis = 2
  We get:
    Out.shape = (3 * 100, 4 * 100)

Case 2:
  Given
    X.shape = (3, 100, 100, 4)
  and
    axis = 0
  We get:
    Out.shape = (1, 3 * 100 * 100 * 4)
)DOC");
  }
};

class FlattenGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    context->SetOutputDim(framework::GradVarName("X"),
                          context->GetInputDim("X"));
    context->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class FlattenGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("flatten_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

// FIXME(zcd): flatten2 adds an intermediate output(XShape) based on flatten,
// the XShape is used to carry the shape and lod of X which will be used in
// flatten_grad, in this way, the framework can reuse the memory of X
// immediately the flatten2_op is finished.
// Considering compatibility issues, we could not fix flatten2_op
class Flatten2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input (X) of Flatten op should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output (Output) of Flatten op should not be null.");
    const auto &axis = ctx->Attrs().Get<int>("axis");
    const auto &in_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(axis, 0,
                      "The axis should be greater than or equal to 0.");
    PADDLE_ENFORCE_LE(
        axis, in_dims.size(),
        "The axis should be less than or equal to input tensor's rank.");

    const auto &out_dims = FlattenOp::GetOutputShape(axis, in_dims);
    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
    if (in_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }

    PADDLE_ENFORCE_EQ(ctx->HasOutput("XShape"), true,
                      "Output (XShape) of Flatten op should not be null.");
    std::vector<int64_t> xshape_dims(in_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < in_dims.size(); ++i) {
      xshape_dims[i + 1] = in_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(xshape_dims));
    ctx->ShareLoD("X", "XShape");
  }
};

class Flatten2OpMaker : public FlattenOpMaker {
 public:
  void Make() override {
    FlattenOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in FlattenGradOp.")
        .AsIntermediate();
  }
};

template <typename T>
class Flatten2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("flatten2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

class Flatten2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE_EQ(context->HasInput("XShape"), true,
                      "Input(XShape) shouldn't be null.");
    PADDLE_ENFORCE_EQ(context->HasInput(framework::GradVarName("Out")), true,
                      "Input(Out@GRAD) shouldn't be null.");
    auto xshape_dims = context->GetInputDim("XShape");
    auto x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    context->SetOutputDim(framework::GradVarName("X"), x_dims);
    context->ShareLoD("XShape", framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

DECLARE_INPLACE_OP_INFERER(FlattenOpInplaceInToOut, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FlattenGradInplaceinToOut,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(FlattenGradNoNeedBufferVarsInference,
                                      "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(flatten, ops::FlattenOp, ops::FlattenOpMaker,
                  ops::FlattenGradOpMaker<paddle::framework::OpDesc>,
                  ops::FlattenGradOpMaker<paddle::imperative::OpBase>,
                  ops::FlattenOpInplaceInToOut);
REGISTER_OPERATOR(flatten_grad, ops::FlattenGradOp,
                  ops::FlattenGradInplaceinToOut,
                  ops::FlattenGradNoNeedBufferVarsInference);

REGISTER_OPERATOR(flatten2, ops::Flatten2Op, ops::Flatten2OpMaker,
                  ops::Flatten2GradOpMaker<paddle::framework::OpDesc>,
                  ops::Flatten2GradOpMaker<paddle::imperative::OpBase>,
                  ops::FlattenOpInplaceInToOut);
REGISTER_OPERATOR(flatten2_grad, ops::Flatten2GradOp,
                  ops::FlattenGradInplaceinToOut);

REGISTER_OP_CPU_KERNEL(
    flatten, ops::FlattenKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FlattenKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FlattenKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FlattenKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::FlattenKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    flatten_grad,
    ops::FlattenGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FlattenGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FlattenGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::FlattenGradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::FlattenGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    flatten2, ops::Flatten2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::Flatten2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::Flatten2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::Flatten2Kernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Flatten2Kernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    flatten2_grad,
    ops::Flatten2GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::Flatten2GradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::Flatten2GradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::Flatten2GradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Flatten2GradKernel<paddle::platform::CPUDeviceContext, int64_t>);
