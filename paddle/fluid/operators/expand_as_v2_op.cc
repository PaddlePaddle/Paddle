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

#include "paddle/fluid/operators/expand_as_v2_op.h"
#include <memory>
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

class ExpandAsV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAsV2");
    OP_INOUT_CHECK(ctx->HasInput("target_tensor"), "Input", "target_tensor",
                   "ExpandAsV2");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ExpandAsV2");
    auto x_dims = ctx->GetInputDim("X");
    auto target_tensor_dims = ctx->GetInputDim("target_tensor");
    PADDLE_ENFORCE_GE(
        target_tensor_dims.size(), static_cast<size_t>(x_dims.size()),
        platform::errors::InvalidArgument(
            "The rank of Input(target_tensor) must be greater than or equal "
            "to the rank of Input(X). But received Input(X): input "
            "rank %u, input shape [%s]; received Input(target_tensor): "
            "input rank %u, input shape [%s].",
            x_dims.size(), x_dims, target_tensor_dims.size(),
            target_tensor_dims));
    PADDLE_ENFORCE_LE(
        target_tensor_dims.size(), MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The rank of Input(target_tensor) must not be less than or equal "
            "to %d. But received: input rank %u, input shape [%s].",
            MAX_RANK_SUPPORTED, x_dims.size(), x_dims));
    std::vector<int64_t> out_shape(target_tensor_dims.size());
    ctx->SetOutputDim("Out", framework::make_ddim(out_shape));
  }
};

class ExpandAsV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
             "X is the input to be expanded.");
    AddOutput("Out",
              "(Tensor, default Tensor<float>). A tensor with rank in [1, 6]."
              "The rank of Output(Out) have the same with Input(X). "
              "After expanding, size of each dimension of Output(Out) is equal "
              "to size of the corresponding dimension of Input(X) multiplying "
              "the corresponding value given by Attr(expand_times).");
    AddInput("target_tensor", "Expand tensor's shape for each dimension.");
    AddComment(R"DOC(
Expand the input by given times number. You should set times
number for each dimension by providing tensor 'expend_tensor'. The rank of X
should be in [1, 6]. Please note that size of 'expend_tensor' must be the same
with X's rank. Following is a using case:
Input(X) is a 3-D tensor with shape [2, 3, 1]:
        [
           [[1], [2], [3]],
           [[4], [5], [6]]
        ]
target_tensors'shape:  [2, 6, 2]
Output(Out) is a 3-D tensor with shape [2, 6, 2]:
        [
            [[1, 1], [2, 2], [3, 3], [1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6], [4, 4], [5, 5], [6, 6]]
        ]
)DOC");
  }
};

class ExpandAsV2GradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "ExpandAsV2Grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "ExpandAsV2Grad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class ExpandAsV2GradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("expand_as_v2_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("target_tensor", this->Input("target_tensor"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ExpandAsV2GradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(expand_as_v2, ops::ExpandAsV2Op, ops::ExpandAsV2OpMaker,
                  ops::ExpandAsV2GradOpMaker<paddle::framework::OpDesc>,
                  ops::ExpandAsV2GradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(expand_as_v2_grad, ops::ExpandAsV2GradOp,
                  ops::ExpandAsV2GradNoNeedBufVarsInferer);
REGISTER_OP_CPU_KERNEL(
    expand_as_v2,
    ops::ExpandAsV2Kernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsV2Kernel<paddle::platform::CPUDeviceContext, double>,
    ops::ExpandAsV2Kernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsV2Kernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsV2Kernel<paddle::platform::CPUDeviceContext, bool>);
REGISTER_OP_CPU_KERNEL(
    expand_as_v2_grad,
    ops::ExpandAsV2GradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ExpandAsV2GradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ExpandAsV2GradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ExpandAsV2GradKernel<paddle::platform::CPUDeviceContext, double>);
