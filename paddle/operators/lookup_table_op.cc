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
#include "paddle/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class LookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(W) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input(Ids) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LookupTableOp should not be null.");

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");

    ctx->SetOutputDim("Out", {ids_dims[0], table_dims[1]});
    ctx->ShareLoD("Ids", /*->*/ "Out");
  }

 protected:
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("W")->type());
  }
};

class LookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LookupTableOpMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("W",
             "An input represents embedding tensors,"
             " which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int64"
             "contains the ids to be looked up in W.");
    AddOutput("Out", "The lookup results, which have the same type with W.");
    AddComment(R"DOC(
This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.

The input `Ids` can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD with input `Ids`.
)DOC");
  }
};

class LookupTableOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto ids_dims = ctx->GetInputDim("Ids");
    auto table_dims = ctx->GetInputDim("W");
    if (dynamic_cast<framework::CompileTimeInferShapeContext*>(ctx)) {
      ctx->SetOutputDim(framework::GradVarName("W"),
                        {ids_dims[0], table_dims[1]});
    } else if (dynamic_cast<framework::RuntimeInferShapeContext*>(ctx)) {
      auto* d_table_var =
          dynamic_cast<framework::RuntimeInferShapeContext*>(ctx)->OutputVar(
              framework::GradVarName("W"));
      auto* sr = d_table_var->GetMutable<framework::SelectedRows>();
      sr->set_height(table_dims[0]);
      framework::Tensor* sr_value = sr->mutable_value();
      sr_value->Resize({ids_dims[0], table_dims[1]});
    }
  }

 protected:
  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("W")->type());
  }
};

class LookupTableOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    auto out_var_name = op_desc.Output(framework::GradVarName("W")).front();
    block->Var(out_var_name)->SetType(framework::VarDesc::SELECTED_ROWS);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lookup_table, ops::LookupTableOp, ops::LookupTableOpMaker);
REGISTER_OPERATOR(lookup_table_grad, ops::LookupTableOpGrad,
                  ops::LookupTableOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(lookup_table, ops::LookupTableKernel<float>);
REGISTER_OP_CPU_KERNEL(lookup_table_grad, ops::LookupTableGradKernel<float>);
