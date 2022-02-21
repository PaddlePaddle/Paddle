// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/tree_conv_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {
class TreeConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("NodesVector",
             "(Tensor) The feature vector of every node on the tree. "
             "The shape of the feature vector must be "
             "[max_tree_node_size, feature_size].");
    AddInput("EdgeSet",
             "(Tensor) The Edges of Tree. The edge must be directional. "
             "The shape of the edge set must be [max_tree_node_size, 2].");
    AddInput("Filter",
             "(Tensor) The feature detector. "
             "The shape of the filter is "
             "[feature_size, 3, output_size, num_filters].");
    AddOutput("Out",
              "(Tensor) The feature vector of subtrees. "
              "The shape of the output tensor is [max_tree_node_size, "
              "output_size, num_filters]. "
              "The output tensor could be a new feature "
              "vector for next tree convolution layers.");
    AddAttr<int>("max_depth",
                 "(int, default: 2) The depth of feature detector.")
        .SetDefault(2)
        .GreaterThan(1);
    AddComment(R"DOC(
**Tree-Based Convolution Operator**

Tree-Based Convolution is a kind of convolution based on tree structure.
Tree-Based Convolution is a part of Tree-Based Convolution Neural Network(TBCNN),
which is used to classify tree structures, such as Abstract Syntax Tree.
Tree-Based Convolution proposed a kind of data structure called continuous binary tree,
which regards multiway tree as binary tree.
The paper of Tree-Based Convolution Operator is here:
https://arxiv.org/abs/1409.5718v1
)DOC");
  }
};
class TreeConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("NodesVector"), "Input", "NodesVector",
                   "TreeConv");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "TreeConv");
    OP_INOUT_CHECK(ctx->HasInput("EdgeSet"), "Input", "EdgeSet", "TreeConv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "TreeConv");

    auto edge_dims = ctx->GetInputDim("EdgeSet");
    auto vector_dims = ctx->GetInputDim("NodesVector");
    auto filter_dims = ctx->GetInputDim("Filter");

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(edge_dims[2], 2,
                        platform::errors::InvalidArgument(
                            "Input(EdgeSet) dim[2] should be 2. "
                            "But received Input(EdgeSet) dim[2] is %d.",
                            edge_dims[2]));
    } else {
      if (edge_dims[2] != -1) {
        PADDLE_ENFORCE_EQ(edge_dims[2], 2,
                          platform::errors::InvalidArgument(
                              "Input(EdgeSet) dim[2] should be 2. "
                              "But received Input(EdgeSet) dim[2] is %d.",
                              edge_dims[2]));
      }
    }
    PADDLE_ENFORCE_EQ(edge_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The dimension of EdgeSet Tensor should be 3. "
                          "But received the dimension of EdgeSet Tensor is %d.",
                          edge_dims.size()));
    PADDLE_ENFORCE_EQ(
        vector_dims.size(), 3,
        platform::errors::InvalidArgument(
            "The dimension of NodesVector Tensor should be 3. "
            "But received the dimension of NodesVector Tensor is %d.",
            vector_dims.size()));
    PADDLE_ENFORCE_EQ(filter_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "The dimension of Filter Tensor should be 4. "
                          "But received the dimension of Filter Tensor is %d.",
                          filter_dims.size()));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(filter_dims[1], 3,
                        platform::errors::InvalidArgument(
                            "Input(Filter) dim[1] should be 3. "
                            "But received Input(Filter) dim[1] is %d.",
                            filter_dims[1]));
      PADDLE_ENFORCE_EQ(
          filter_dims[0], vector_dims[2],
          platform::errors::InvalidArgument(
              "Input(Filter) dim[0] must equal to Input(NodesVector) dim[2]. "
              "But received Input(Filter) dim[0] = %d, Input(NodesVector) "
              "dim[2] = %d.",
              filter_dims[0], vector_dims[2]));
    } else {
      if (filter_dims[1] != -1) {
        PADDLE_ENFORCE_EQ(filter_dims[1], 3,
                          platform::errors::InvalidArgument(
                              "Input(Filter) dim[1] should be 3. "
                              "But received Input(Filter) dim[1] is %d.",
                              filter_dims[1]));
      }

      if (filter_dims[0] != -1 && vector_dims[2] != -1) {
        PADDLE_ENFORCE_EQ(
            filter_dims[0], vector_dims[2],
            platform::errors::InvalidArgument(
                "Input(Filter) dim[0] must equal to Input(NodesVector) dim[2]. "
                "But received Input(Filter) dim[0] = %d, Input(NodesVector) "
                "dim[2] = %d.",
                filter_dims[0], vector_dims[2]));
      }
    }
    auto output_dims = phi::make_ddim(
        {vector_dims[0], vector_dims[1], filter_dims[2], filter_dims[3]});
    ctx->SetOutputDim("Out", output_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "NodesVector"),
        ctx.device_context());
  }
};

template <typename T>
class TreeConvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("tree_conv_grad");

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput("EdgeSet", this->Input("EdgeSet"));
    op->SetInput("NodesVector", this->Input("NodesVector"));

    op->SetOutput(framework::GradVarName("NodesVector"),
                  this->InputGrad("NodesVector"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));

    op->SetAttrMap(this->Attrs());
  }
};

class TreeConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "grad_TreeConv");
    OP_INOUT_CHECK(ctx->HasInput("EdgeSet"), "Input", "EdgeSet",
                   "grad_TreeConv");
    OP_INOUT_CHECK(ctx->HasInput("NodesVector"), "Input", "NodesVector",
                   "grad_TreeConv");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "grad_TreeConv");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("NodesVector")),
                   "Output", framework::GradVarName("NodesVector"),
                   "grad_TreeConv");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Filter")), "Output",
                   framework::GradVarName("Filter"), "grad_TreeConv");

    auto vectors_dims = ctx->GetInputDim("NodesVector");
    auto filter_dims = ctx->GetInputDim("Filter");
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("NodesVector"))) {
      ctx->SetOutputDim(framework::GradVarName("NodesVector"), vectors_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "NodesVector"),
        ctx.device_context());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(tree_conv, ops::TreeConvOp, ops::TreeConvOpMaker,
                  ops::TreeConvGradOpMaker<paddle::framework::OpDesc>,
                  ops::TreeConvGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(tree_conv_grad, ops::TreeConvGradOp);

REGISTER_OP_CPU_KERNEL(
    tree_conv, ops::TreeConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TreeConvKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    tree_conv_grad,
    ops::TreeConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TreeConvGradKernel<paddle::platform::CPUDeviceContext, double>);
