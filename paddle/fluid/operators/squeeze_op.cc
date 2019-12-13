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

#include "paddle/fluid/operators/squeeze_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class SqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of Squeeze operator should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of Squeeze operator should not be null.");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE_LE(x_dims.size(), 6,
                      "ShapeError: the dimensions of Input(X) "
                      "should be in the range of [1, 6] (Eigen limit)."
                      "But received X's dimensions = %d, X's shape=[%s].",
                      x_dims.size(), x_dims);

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    for (int a : axes) {
      PADDLE_ENFORCE_LT(
          a, x_dims.size(),
          "ShapeError: The squeeze axis should be less than input "
          "tensor's dimensions. But received axis = %d, input "
          "tensor's dimensions = %d, input tensor's shape = [%s].",
          a, x_dims.size(), x_dims);
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
  }

  static framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                                        const framework::DDim &in_dims) {
    size_t num_squeeze_dims = squeeze_dims.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};

    // Determines number of dimensions of output tensor after squeeze.
    // Mark and count the dimensions need to be squeezed
    if (num_squeeze_dims == 0) {
      for (int idx = 0; idx < in_dims.size(); ++idx) {
        if (in_dims[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (size_t idx = 0; idx < num_squeeze_dims; ++idx) {
        int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                            : squeeze_dims[idx];
        PADDLE_ENFORCE_GE(current, 0,
                          "Invalid axis, the axis should >= 0."
                          "Current axis is:%d, input tensor's shape = [%s].",
                          current, in_dims);

        if (!(should_squeeze[current])) {
          ++cnt_squeezed_dims;
        }
        should_squeeze[current] = true;
      }
    }

    // Make output dimensions
    std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
    for (int in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_shape[out_idx++] = in_dims[in_idx];
      }
    }

    return framework::make_ddim(output_shape);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class SqueezeGradOp : public framework::OperatorWithKernel {
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

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of squeeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of squeeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of integers,"
                              " indicating the dimensions to squeeze.")
        .SetDefault({});
    AddComment(R"DOC(
        Squeeze Operator.

        Remove single-dimensional entries from the shape of a tensor.
        Takes a parameter axes with a list of axes to squeeze.
        If axes is not provided, all the single dimensions will be removed from the shape.
        If an axis is selected with shape entry not equal to one, an error is raised.

        Examples:
        Case 1:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = [0]
          we get:
            Out.shape = (3, 1, 5)

        Case 2:
          Given
            X.shape = (1, 3, 1, 5)
          and
            axes = []
          we get:
            Out.shape = (3, 5)
    )DOC");
  }
};

class Squeeze2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      "Input(X) of Squeeze operator should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of Squeeze operator should not be null.");

    const auto &x_dims = ctx->GetInputDim("X");
    // Check input tensor dims (<6) Eigen limit.
    PADDLE_ENFORCE_LE(x_dims.size(), 6,
                      "ShapeError: the dimensions of Input(X) "
                      "should be in the range of [1, 6] (Eigen limit)."
                      "But received X's dimensions = %d, X's shape = [%s].",
                      x_dims.size(), x_dims);

    const auto &axes = ctx->Attrs().Get<std::vector<int>>("axes");
    for (int a : axes) {
      PADDLE_ENFORCE_LT(
          a, x_dims.size(),
          "ShapeError: The squeeze axis should be less than input "
          "tensor's dimensions. But received axis = %d, input "
          "tensor's dimensions = %d, input tensor's shape = [%s].",
          a, x_dims.size(), x_dims);
    }

    auto out_dims = SqueezeOp::GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }

    PADDLE_ENFORCE_EQ(ctx->HasOutput("XShape"), true,
                      "Output(XShape) of Squeeze operator should not be null.");
    std::vector<int64_t> xshape_dims(x_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < x_dims.size(); ++i) {
      xshape_dims[i + 1] = x_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(xshape_dims));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }
};

template <typename T>
class SqueezeGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("squeeze_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

class Squeeze2GradOp : public framework::OperatorWithKernel {
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

// FIXME(zcd): squeeze2 adds an intermediate output(XShape) based on squeeze,
// the XShape is used to carry the shape and lod of X which will be used in
// squeeze_grad, in this way, the framework can reuse the memory of X
// immediately the squeeze2_op is finished.
// Considering compatibility issues, we could not fix squeeze2_op
class Squeeze2OpMaker : public SqueezeOpMaker {
 public:
  void Make() override {
    SqueezeOpMaker::Make();
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in SqueezeGradOp.")
        .AsIntermediate();
  }
};

template <typename T>
class Squeeze2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto *grad_op = new T();
    grad_op->SetType("squeeze2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

DECLARE_INPLACE_OP_INFERER(SequeezeInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(SequeezeGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(SqueezeGradNoNeedBufferVarsInference,
                                      "X");
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(squeeze, ops::SqueezeOp, ops::SqueezeOpMaker,
                  ops::SqueezeGradOpMaker<paddle::framework::OpDesc>,
                  ops::SqueezeGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(squeeze_grad, ops::SqueezeGradOp,
                  ops::SqueezeGradNoNeedBufferVarsInference);

REGISTER_OPERATOR(squeeze2, ops::Squeeze2Op, ops::Squeeze2OpMaker,
                  ops::Squeeze2GradOpMaker<paddle::framework::OpDesc>,
                  ops::Squeeze2GradOpMaker<paddle::imperative::OpBase>,
                  ops::SequeezeInplaceInferer);
REGISTER_OPERATOR(squeeze2_grad, ops::Squeeze2GradOp,
                  ops::SequeezeGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    squeeze, ops::SqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze2, ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Squeeze2Kernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze2_grad,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::Squeeze2GradKernel<paddle::platform::CPUDeviceContext, int64_t>);
