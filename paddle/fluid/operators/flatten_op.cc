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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {
// FIXME(zcd): flatten2 adds an intermediate output(XShape) based on flatten,
// the XShape is used to carry the shape and lod of X which will be used in
// flatten_grad, in this way, the framework can reuse the memory of X
// immediately the flatten2_op is finished.
// Considering compatibility issues, we could not fix flatten2_op
class Flatten2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Flatten2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Flatten2");
    const auto &axis = ctx->Attrs().Get<int>("axis");
    const auto &in_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(axis,
                      0,
                      phi::errors::InvalidArgument(
                          "The axis should be greater than or equal to 0."));
    PADDLE_ENFORCE_LE(
        axis,
        in_dims.size(),
        phi::errors::InvalidArgument(
            "The axis should be less than or equal to input tensor's rank"));

    const auto &out_dims = Flatten2Op::GetOutputShape(axis, in_dims);
    ctx->SetOutputDim("Out", common::make_ddim(out_dims));
    if (in_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", "Out");
    }
    if (!ctx->HasOutput("XShape")) return;
    // OP_INOUT_CHECK(ctx->HasOutput("XShape"), "Output", "XShape", "Flatten2");
    std::vector<int64_t> xshape_dims(in_dims.size() + 1);
    xshape_dims[0] = 0;
    for (int i = 0; i < in_dims.size(); ++i) {
      xshape_dims[i + 1] = in_dims[i];
    }
    ctx->SetOutputDim("XShape", common::make_ddim(xshape_dims));
    ctx->ShareLoD("X", "XShape");
  }

  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }

  static std::vector<int32_t> GetOutputShape(const int axis,
                                             const framework::DDim &in_dims) {
    if (in_dims.size() == 0) {
      return {1};
    }

    int64_t outer = 1, inner = 1;
    for (int i = 0; i < in_dims.size(); ++i) {
      if (i < axis) {
        if (in_dims[i] == -1 || outer == -1) {
          outer = -1;
        } else {
          outer *= in_dims[i];
        }
      } else {
        if (in_dims[i] == -1 || inner == -1) {
          inner = -1;
        } else {
          inner *= in_dims[i];
        }
      }
    }
    std::vector<int32_t> out_shape(2);
    out_shape[0] = static_cast<int32_t>(outer);
    out_shape[1] = static_cast<int32_t>(inner);
    return out_shape;
  }
};

class Flatten2OpMaker : public framework::OpProtoAndCheckerMaker {
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
    AddOutput("XShape",
              "XShape is just used to store the shape and lod of X, which will "
              "be used in FlattenGradOp.")
        .AsIntermediate()
        .AsExtra();
  }
};

template <typename T>
class Flatten2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("flatten2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Flatten2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(
        context->HasInput("XShape"), "Input", "XShape", "Flatten2Grad");
    OP_INOUT_CHECK(context->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "Flatten2Grad");
    auto xshape_dims = context->GetInputDim("XShape");
    auto x_dims = common::slice_ddim(xshape_dims, 1, xshape_dims.size());
    context->SetOutputDim(framework::GradVarName("X"), x_dims);
    context->ShareLoD("XShape", framework::GradVarName("X"));
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return phi::KernelKey(input_data_type, ctx.GetPlace());
  }
};

DECLARE_INPLACE_OP_INFERER(FlattenOpInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(FlattenGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});
DECLARE_NO_NEED_BUFFER_VARS_INFERER(FlattenGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(flatten2,
                  ops::Flatten2Op,
                  ops::Flatten2OpMaker,
                  ops::Flatten2GradOpMaker<paddle::framework::OpDesc>,
                  ops::Flatten2GradOpMaker<paddle::imperative::OpBase>,
                  ops::FlattenOpInplaceInferer);
REGISTER_OPERATOR(flatten2_grad,
                  ops::Flatten2GradOp,
                  ops::FlattenGradInplaceInferer);

REGISTER_OP_CPU_KERNEL(flatten2,
                       ops::Flatten2Kernel<phi::CPUContext, float>,
                       ops::Flatten2Kernel<phi::CPUContext, double>,
                       ops::Flatten2Kernel<phi::CPUContext, uint8_t>,
                       ops::Flatten2Kernel<phi::CPUContext, int>,
                       ops::Flatten2Kernel<phi::CPUContext, int8_t>,
                       ops::Flatten2Kernel<phi::CPUContext, int64_t>);
REGISTER_OP_CPU_KERNEL(flatten2_grad,
                       ops::Flatten2GradKernel<phi::CPUContext, float>,
                       ops::Flatten2GradKernel<phi::CPUContext, double>,
                       ops::Flatten2GradKernel<phi::CPUContext, uint8_t>,
                       ops::Flatten2GradKernel<phi::CPUContext, int>,
                       ops::Flatten2GradKernel<phi::CPUContext, int8_t>,
                       ops::Flatten2GradKernel<phi::CPUContext, int64_t>);
