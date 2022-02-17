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

#include "paddle/fluid/operators/scatter_nd_add_op.h"
#include <memory>
#include <vector>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class ScatterNdAddOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of ScatterNdAddOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Index"), true,
        platform::errors::InvalidArgument(
            "Input(Index) of ScatterNdAddOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Updates"), true,
        platform::errors::InvalidArgument(
            "Input(Updates) of ScatterNdAddOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of ScatterNdAddOp should not be null."));

    auto ref_dims = ctx->GetInputDim("X");
    auto ref_dims_size = ref_dims.size();
    auto index_dims = ctx->GetInputDim("Index");
    auto index_dims_size = index_dims.size();
    auto updates_dims = ctx->GetInputDim("Updates");
    auto updates_dims_size = updates_dims.size();

    PADDLE_ENFORCE_LE(
        index_dims[index_dims_size - 1], ref_dims_size,
        platform::errors::InvalidArgument(
            "The last dimension of Input(Index)'s shape should be no greater "
            "than the rank of Input(X), but received the last dimension of "
            "Input(Index)'s shape is %d, the rank of Input(X) is %d.",
            index_dims[index_dims_size - 1], ref_dims_size));
    PADDLE_ENFORCE_GE(index_dims_size, 2UL,
                      platform::errors::InvalidArgument(
                          "The rank of Input(Index) should be greater than 1, "
                          "but received the rank of Input(Index) is %d.",
                          index_dims_size));

    // update.shape = index.shape[:-1] + output.shape[index.shape[-1]:]
    std::vector<int64_t> r_updates_dims;
    for (int64_t i = 0; i < index_dims_size - 1; ++i) {
      r_updates_dims.emplace_back(index_dims[i]);
    }
    for (int64_t i = index_dims[index_dims_size - 1]; i < ref_dims_size; ++i) {
      r_updates_dims.emplace_back(ref_dims[i]);
    }

    PADDLE_ENFORCE_EQ(
        r_updates_dims.size(), updates_dims_size,
        platform::errors::InvalidArgument(
            "Updates has wrong shape. The shape of Updates and Input(Updates) "
            "should be same, but received the shape of Updates is %d, "
            "the shape of Input(Updates) is %d.",
            r_updates_dims.size(), updates_dims_size));

    for (int64_t i = 0; i < updates_dims_size; ++i) {
      PADDLE_ENFORCE_EQ(
          r_updates_dims[i], updates_dims[i],
          platform::errors::InvalidArgument(
              "Updates has wrong shape. The dimensions of Updates and "
              "Input(Updates) should match, but received Updates's"
              "%d-th dimension is %d, Input(Updates)'s %d-th "
              "dimension is %d.",
              i, r_updates_dims[i], i, updates_dims[i]));
    }
    ctx->SetOutputDim("Out", ref_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                      OperatorWithKernel::IndicateVarDataType(ctx, "Updates"),
                      platform::errors::InvalidArgument(
                          "Ref and Updates must have same type"));
    return framework::OpKernelType(
        framework::TransToProtoVarType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ScatterNdAddGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("Updates"))) {
      ctx->SetOutputDim(framework::GradVarName("Updates"),
                        ctx->GetInputDim("Updates"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"),
                        ctx->GetInputDim(framework::GradVarName("Out")));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class ScatterNdAddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of scatter_nd_add op");
    AddInput("Index",
             "The index input of scatter_nd_add op where X will be updated");
    AddInput("Updates", "The updated value of scatter_nd_add op");
    AddOutput("Out", "The output of scatter_nd_add op");
    AddComment(R"DOC(
Scatter_nd_add Operator.

Output is obtained by applying sparse addition to a single value or slice in a Variable.

      Given:
        * Case 1:
            ref = [0, 1, 2, 3, 4, 5]
            index = [[1], [2], [3], [1]]
            updates = [9, 10, 11, 12]

          we get:

            output = [0, 22, 12, 14, 4, 5]

        * Case 2:
            ref = [[65, 17], [-14, -25]]
            index = [[], []]
            updates = [[[-1, -2], [1, 2]],
                       [[3, 4], [-3, -4]]]
            ref.shape = (2, 2)
            index.shape = (2, 0)
            updates.shape = (2, 2, 2)

          we get:

            output = [[67, 19], [-16, -27]]
)DOC");
  }
};

template <typename T>
class ScatterNdAddGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("scatter_nd_add_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("Updates", this->Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"),
                  this->InputGrad("Updates"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(ScatterNdAddGradNoNeedBufferVarsInferer,
                                    "Updates");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(scatter_nd_add, ops::ScatterNdAddOp, ops::ScatterNdAddOpMaker,
                  ops::ScatterNdAddGradMaker<paddle::framework::OpDesc>,
                  ops::ScatterNdAddGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(scatter_nd_add_grad, ops::ScatterNdAddGradOp,
                  ops::ScatterNdAddGradNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(scatter_nd_add, ops::ScatterNdAddOpKernel<float>,
                       ops::ScatterNdAddOpKernel<double>,
                       ops::ScatterNdAddOpKernel<int64_t>,
                       ops::ScatterNdAddOpKernel<int>,
                       ops::ScatterNdAddOpKernel<uint8_t>);

REGISTER_OP_CPU_KERNEL(scatter_nd_add_grad,
                       ops::ScatterNdAddGradientOpKernel<float>,
                       ops::ScatterNdAddGradientOpKernel<double>,
                       ops::ScatterNdAddGradientOpKernel<int64_t>,
                       ops::ScatterNdAddGradientOpKernel<int>,
                       ops::ScatterNdAddGradientOpKernel<uint8_t>);
