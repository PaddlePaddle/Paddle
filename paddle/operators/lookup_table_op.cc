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

#include "paddle/operators/lookup_table_op.h"

namespace paddle {
namespace operators {

class LookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContextBase &ctx) const override {
    PADDLE_ENFORCE(ctx.HasInput("W"),
                   "Input(W) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx.HasInput("Ids"),
                   "Input(Ids) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx.HasOutput("Out"),
                   "Output(Out) of LookupTableOp should not be null.");

    auto table_dims = ctx.GetInputDim("W");
    auto ids_dims = ctx.GetInputDim("Ids");
    auto output_dims = ctx.GetOutputDim("Out");

    ctx.SetOutputDim("Out", {ids_dims[0], table_dims[1]});
  }
};

class LookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LookupTableOpMaker(framework::OpProto *proto,
                     framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("W",
             "An input represents embedding tensors,"
             " which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64"
             "contains the ids to be looked up in W.");
    AddOutput("Out", "The lookup results, which have the same type with W.");
    AddComment(
        "This operator is used to perform lookups on the parameter W,"
        "then concatenated into a dense tensor.");
  }
};

class LookupTableOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &context) const override {
    auto table = context.Input<Tensor>("W");
    auto d_table =
        context.Output<framework::LoDTensor>(framework::GradVarName("W"));
    d_table->Resize(table->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lookup_table, ops::LookupTableOp, ops::LookupTableOpMaker,
            lookup_table_grad, ops::LookupTableOpGrad);

REGISTER_OP_CPU_KERNEL(lookup_table, ops::LookupTableKernel<float>);
REGISTER_OP_CPU_KERNEL(lookup_table_grad, ops::LookupTableGradKernel<float>);
