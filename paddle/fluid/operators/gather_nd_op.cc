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

#include "paddle/fluid/operators/gather_nd_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class GatherNdOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of GatherNdOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Index"), true,
                      platform::errors::InvalidArgument(
                          "Input(Index) of GatherNdOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of GatherNdOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    auto x_dims_size = x_dims.size();
    auto index_dims = ctx->GetInputDim("Index");
    auto index_dims_size = index_dims.size();

    PADDLE_ENFORCE_LE(
        index_dims[index_dims_size - 1], x_dims_size,
        platform::errors::InvalidArgument(
            "Input(Index).shape[-1] should be no greater than Input(X).rank"));
    PADDLE_ENFORCE_GE(index_dims_size, 2UL,
                      platform::errors::InvalidArgument(
                          "The rank of Input(Index) should be greater than 1"));

    std::vector<int64_t> result_dims;
    // The result dims is
    //   Index.shape[:-1] + X.shape[Index.shape[-1]:]
    for (int i = 0; i < index_dims_size - 1; ++i) {
      result_dims.emplace_back(index_dims[i]);
    }
    for (int i = index_dims[index_dims_size - 1]; i < x_dims_size; ++i) {
      result_dims.emplace_back(x_dims[i]);
    }

    ctx->SetOutputDim("Out", framework::make_ddim(result_dims));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    const auto& x_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(
        x_type,
        x_type == framework::proto::VarType::BOOL
            ? x->place()  // to be consistent with compare and logical ops
            : ctx.device_context().GetPlace());
  }
};

class GatherNdGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*-->*/ framework::GradVarName("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class GatherNdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of gather_nd op");
    AddInput("Index", "The index input of gather_nd op");
    AddOutput("Out", "The output of gather_nd op");
    AddComment(R"DOC(
    Gather_Nd Operator.

    This function is actually a high-dimensional extension of gather 
    and supports for simultaneous indexing by multiple axes. Out is 
    obtained by gathering slices from X into a tensor with shape 
    Index.shape[:-1] + X.shape[Index.shape[-1]:].

    Example:
   
    Given:
         X = [[[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]],
              [[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]]]
       
         X.shape = (2, 3, 4)

   *Case 1:

       Index = [[1]]

    we get:
       Out = 
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]

   *Case 2:

       Index = [[0,2]]

    we get:
        
       Out =  [8, 9, 10, 11]

   *Case 3:

       Index = [[1, 2, 3]]

    we get:

       Out = [23]

)DOC");
  }
};

template <typename T>
class GatherNdGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gather_nd_grad");
    op->SetInput("Index", this->Input("Index"));
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GatherNdGradNoNeedBufferVarInference, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gather_nd, ops::GatherNdOp, ops::GatherNdOpMaker,
                  ops::GatherNdGradOpMaker<paddle::framework::OpDesc>,
                  ops::GatherNdGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(gather_nd_grad, ops::GatherNdGradOp,
                  ops::GatherNdGradNoNeedBufferVarInference);

REGISTER_OP_CPU_KERNEL(gather_nd, ops::GatherNdOpKernel<float>,
                       ops::GatherNdOpKernel<double>,
                       ops::GatherNdOpKernel<int64_t>,
                       ops::GatherNdOpKernel<int>, ops::GatherNdOpKernel<bool>,
                       ops::GatherNdOpKernel<uint8_t>);

REGISTER_OP_CPU_KERNEL(gather_nd_grad, ops::GatherNdGradOpKernel<float>,
                       ops::GatherNdGradOpKernel<double>,
                       ops::GatherNdGradOpKernel<int64_t>,
                       ops::GatherNdGradOpKernel<int>,
                       ops::GatherNdGradOpKernel<uint8_t>);
