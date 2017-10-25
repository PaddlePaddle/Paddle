/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/sum_op.h"
#include <vector>
#include "paddle/framework/var_type_inference.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace operators {
using framework::Tensor;

class SumOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("X"), "Inputs(X) should not be null");
    auto x_dims = ctx->GetInputsDim("X");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SumOp should not be null.");

    size_t N = x_dims.size();
    PADDLE_ENFORCE_GT(N, 1, "Input tensors count should > 1.");

    bool is_internal = ctx->Attrs().Get<bool>("is_internal");
    if (is_internal) {
      if (dynamic_cast<framework::CompileTimeInferShapeContext*>(ctx)) {
        auto in_dim = x_dims[0];
        ctx->SetOutputDim("Out", in_dim);
      } else if (dynamic_cast<framework::RuntimeInferShapeContext*>(ctx)) {
        // Runtime InferShape InferShape is done in Compute method.
      }
    } else {
      auto in_dim = x_dims[0];
      for (size_t i = 1; i < N; i++) {
        auto dim = x_dims[i];
        PADDLE_ENFORCE_EQ(in_dim, dim, "Input tensors must have same shape");
      }
      ctx->SetOutputDim("Out", in_dim);
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }
};

class SumOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SumOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "the input tensors of sum operator.").AsDuplicable();
    AddOutput("Out", "the output tensor of sum operator.");
    AddAttr<bool>("is_internal", "Used internally when suming Gradient")
        .SetDefault(false);
    AddComment(R"DOC(
Sum the input tensors.

All the inputs can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with the first input.
)DOC");
  }
};

class SumOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    auto& inputs = op_desc.Input("X");
    auto default_var_type = framework::VarDesc::SELECTED_ROWS;

    bool any_input_is_lod_tensor = std::any_of(
        inputs.begin(), inputs.end(), [block](const std::string& name) {
          return block->Var(name)->GetType() == framework::VarDesc::LOD_TENSOR;
        });
    if (any_input_is_lod_tensor) {
      default_var_type = framework::VarDesc::LOD_TENSOR;
    }

    auto out_var_name = op_desc.Output("Out").front();
    block->Var(out_var_name)->SetType(default_var_type);
  }
};

class SumGradMaker : public framework::GradOpDescMakerBase {
 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDescBind>> operator()()
      const override {
    auto x_grads = InputGrad("X");
    std::vector<std::unique_ptr<framework::OpDescBind>> grad_ops;
    grad_ops.reserve(x_grads.size());
    auto og = OutputGrad("Out");
    std::transform(x_grads.begin(), x_grads.end(), std::back_inserter(grad_ops),
                   [&og](const std::string& x_grad) {
                     auto* grad_op = new framework::OpDescBind();
                     grad_op->SetType("scale");
                     grad_op->SetInput("X", og);
                     grad_op->SetOutput("Out", {x_grad});
                     grad_op->SetAttr("scale", 1.0f);
                     return std::unique_ptr<framework::OpDescBind>(grad_op);
                   });
    return grad_ops;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(sum, ops::SumOp, ops::SumOpMaker, ops::SumGradMaker,
                  ops::SumOpVarTypeInference);
REGISTER_OP_CPU_KERNEL(sum, ops::SumKernel<paddle::platform::CPUPlace, float>);
