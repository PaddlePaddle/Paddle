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

#include "paddle/fluid/operators/hierarchical_sigmoid_op.h"
#include <string>
#include <vector>
namespace paddle {
namespace operators {

/**
 * Organize the classes into a binary tree. At each node, a sigmoid function
 * is used to calculate the probability of belonging to the right branch.
 * This idea is from "F. Morin, Y. Bengio (AISTATS 05):
 * Hierarchical Probabilistic Neural Network Language Model."
 *
 * Here we uses a simple way of making the binary tree.
 * Assuming the number of classes C = 6,
 * The classes are organized as a binary tree in the following way:
 *
 * @code{.py}
 * *-*-*- 2
 * | | |- 3
 * | |
 * | |-*- 4
 * |   |- 5
 * |
 * |-*- 0
 *   |- 1
 * @endcode
 *
 * where * indicates an internal node, and each leaf node represents a class.
 * - Node 0 ... C-2 are internal nodes.
 * - Node C-1 ... 2C-2 are leaf nodes.
 * - Class c is represented by leaf node \f$c+C-1\f$.
 *
 * We assign an id for each node:
 * - the id of root be 0.
 * - the left child of a node i is 2*i+1.
 * - the right child of a node i is 2*i+2.
 *
 * It's easy to see that:
 * - the parent of node i is \f$\left\lfloor(i-1)/2\right\rfloor\f$.
 * - the j-th level ancestor of node i is
 * \f$\left\lfloor(i+1)/2^{j+1}\right\rfloor - 1\f$.
 * - A node i is a left child of its parent if \f$(i-1)\%2==0\f$.
 *
 */

class HierarchicalSigmoidOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of HierarchicalSigmoidOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Label"), true,
        platform::errors::NotFound(
            "Input(Label) of HierarchicalSigmoidOp is not found."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      platform::errors::NotFound(
                          "Input(W) of HierarchicalSigmoidOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::NotFound(
            "Output(Out) of HierarchicalSigmoidOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("PreOut"), true,
        platform::errors::NotFound(
            "Output(PreOut) of HierarchicalSigmoidOp is not found."));
    auto with_prefetch = ctx->Attrs().Get<bool>("remote_prefetch");
    if (with_prefetch) {
      PADDLE_ENFORCE_EQ(
          ctx->HasOutput("W_Out"), true,
          platform::errors::NotFound(
              "Output(W_Out) of HierarchicalSigmoidOp is not found."));
    }
    const int64_t batch_size = ctx->GetInputDim("X")[0];
    std::vector<int64_t> output_shape({batch_size, 1});
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

/*
 * Inputs: X, W, Label, PathTable, PathCode, Bias
 * Outputs: Out, PreOut, W_out
 */
template <typename AttrType>
class HierarchicalSigmoidOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, required) The input tensor with shape [N, D], "
             "where N is the size of mini-batch, and D is the feature size.");
    AddInput("W",
             "(LoDTensor, required), The parameters of hierarchical "
             "sigmoid operator, each of them is a 2-D tensor, the shape is"
             "[K, D]. Which K is the num of non-leaf node in Path Tree");
    AddInput("Label",
             "(LoDTensor, required), The labels of training data. It's a"
             "tensor with shape [N, 1].");
    AddInput("PathTable",
             "(LoDTensor, optional), The Path Table from root to current word"
             "it should have shape like [N, L], L is the length of the Path")
        .AsDispensable();
    AddInput(
        "PathCode",
        "(LoDTensor, optional), The Code on each Node of the Path from root "
        "to current word"
        "it should have shape like [N, L], L is the length of the Path")
        .AsDispensable();
    AddInput("Bias",
             "(LoDTensor, optional), The bias is a tensor with shape or "
             "[num_classes, 1]"
             "[num_classes - 1, 1].")
        .AsDispensable();
    AddOutput(
        "Out",
        "(LoDTensor, required) The output of hierarchical sigmoid operator."
        "The shape is [N, 1].");
    AddOutput("PreOut",
              "(LoDTensor, required) A intermedia 2-D tensor with shape "
              "[batch_size, code_length], where code_length represents the "
              "maximum path length from root to leaf nodes.")
        .AsIntermediate();
    AddOutput(
        "W_Out",
        "(LoDTensor, optional) using input 'W' as Output to make it mutable"
        "When we are using prefetch")
        .AsIntermediate();
    AddAttr<AttrType>("num_classes", "(int, optional), The number of classes")
        .SetDefault(2);
    // for parameter prefetch
    AddAttr<bool>("remote_prefetch", "").SetDefault(false);
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));
    AddAttr<std::vector<std::string>>(
        "epmap",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, the split table names that will be fetched from "
        "parameter server)"
        "in the order of input variables for mapping")
        .SetDefault({});
    AddComment(R"DOC(
The hierarchical sigmoid operator organize the classes into a binary tree.
At each node, a sigmoid function is used to calculate the probability of
belonging to the right branch. This idea is from
"F. Morin, Y. Bengio (AISTATS 05):
Hierarchical Probabilistic Neural Network Language Model."
      )DOC");
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
  }
};

/*
 * Inputs: X, W, Label, PathTable, PathCode, PreOut, Out@GRAD
 * Outputs: X@GRAD, W@GRAD, Bias@GRAD
 */
template <typename T>
class HierarchicalSigmoidGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  std::unique_ptr<T> Apply() const override {
    auto* op = new T();
    op->SetType(this->ForwardOpType() + "_grad");
    // Inputs: X, W, Label, PathTable, PathCode, PreOut, Out@GRAD
    op->SetInput("X", this->Input("X"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput("PathTable", this->Input("PathTable"));
    op->SetInput("PathCode", this->Input("PathCode"));
    op->SetInput("PreOut", this->Output("PreOut"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    // Outputs: X@GRAD, W@GRAD, Bias@GRAD
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetAttrMap(this->Attrs());

    return std::unique_ptr<T>(op);
  }
};

class HierarchicalSigmoidGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::NotFound(
            "Input(W) of HierarchicalSigmoidGradOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Label"), true,
        platform::errors::NotFound(
            "Input(Label) of HierarchicalSigmoidGradOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::NotFound(
            "Input(Out@Grad) of HierarchicalSigmoidGradOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("PreOut"), true,
        platform::errors::NotFound(
            "Input(Preout) of HierarchicalSigmoidGradOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("W")), true,
        platform::errors::NotFound(
            "Output(W@Grad of HierarchicalSigmoidGradOp is not found."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput(framework::GradVarName("X")), true,
        platform::errors::NotFound(
            "Output(X@Grad of HierarchicalSigmoidGradOp is not found."));

    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
    }
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class HierarchicalSigmoidGradOpGradVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    auto w_grad_var_name = ctx->Output(framework::GradVarName("W")).front();
    auto has_bias_grad_var = ctx->HasOutput(framework::GradVarName("Bias"));
    std::string bias_grad_var_name;
    bool hasBias = false;
    if (has_bias_grad_var) {
      hasBias = true;
      bias_grad_var_name = ctx->Output(framework::GradVarName("Bias")).front();
    }
    auto attr = ctx->GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "hierarchical_sigmoid_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      ctx->SetType(w_grad_var_name, framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "hierarchical_sigmoid_grad op " << framework::GradVarName("W")
              << " is set to LoDTensor";
      ctx->SetType(w_grad_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    if (hasBias) {
      VLOG(3) << "hierarchical_sigmoid_grad op "
              << framework::GradVarName("Bias") << " is set to LoDTensor";
      ctx->SetType(bias_grad_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetDataType(w_grad_var_name, ctx->GetDataType(ctx->Input("W")[0]));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(
    HierarchicalSigmoidGradOpNoNeedBufferVarInference, "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    hierarchical_sigmoid, ops::HierarchicalSigmoidOp,
    ops::HierarchicalSigmoidOpMaker<int>,
    ops::HierarchicalSigmoidGradMaker<paddle::framework::OpDesc>,
    ops::HierarchicalSigmoidGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(hierarchical_sigmoid_grad, ops::HierarchicalSigmoidGradOp,
                  ops::HierarchicalSigmoidGradOpGradVarTypeInference,
                  ops::HierarchicalSigmoidGradOpNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(
    hierarchical_sigmoid,
    ops::HierarchicalSigmoidOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::HierarchicalSigmoidOpKernel<paddle::platform::CPUDeviceContext,
                                     double>);
REGISTER_OP_CPU_KERNEL(
    hierarchical_sigmoid_grad,
    ops::HierarchicalSigmoidGradOpKernel<paddle::platform::CPUDeviceContext,
                                         float>,
    ops::HierarchicalSigmoidGradOpKernel<paddle::platform::CPUDeviceContext,
                                         double>);
