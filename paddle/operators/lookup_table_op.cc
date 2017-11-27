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

    PADDLE_ENFORCE_EQ(ids_dims.size(), 2);
    PADDLE_ENFORCE_EQ(ids_dims[1], 1);

    ctx->SetOutputDim("Out", {ids_dims[0], table_dims[1]});
    ctx->ShareLoD("Ids", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<LoDTensor>("W")->type()),
        ctx.device_context());
  }
};

class LookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LookupTableOpMaker(framework::OpProto* proto,
                     framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("W",
             "An input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "An input with type int32 or int64 "
             "contains the ids to be looked up in W. "
             "Ids must be a column vector with rank = 2. "
             "The 2nd dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update")
        .SetDefault(false);
    AddComment(R"DOC(
Lookup Table Operator.

This operator is used to perform lookups on the parameter W,
then concatenated into a dense tensor.

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

class LookupTableOpGradDescMaker
    : public framework::DefaultGradOpDescMaker<true> {
  using ::paddle::framework::DefaultGradOpDescMaker<
      true>::DefaultGradOpDescMaker;

 protected:
  virtual std::string GradOpType() const { return "lookup_table_grad"; }
};

class LookupTableOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto table_dims = ctx->GetInputDim("W");
    ctx->SetOutputDim(framework::GradVarName("W"), table_dims);
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<LoDTensor>("W")->type()),
        ctx.device_context());
  }
};

class LookupTableOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind& op_desc,
                  framework::BlockDescBind* block) const override {
    auto out_var_name = op_desc.Output(framework::GradVarName("W")).front();
    auto attr = op_desc.GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      block->Var(out_var_name)->SetType(framework::VarDesc::SELECTED_ROWS);
    } else {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to LoDTensor";
      block->Var(out_var_name)->SetType(framework::VarDesc::LOD_TENSOR);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lookup_table, ops::LookupTableOp,
                  ops::LookupTableOpGradDescMaker, ops::LookupTableOpMaker);
REGISTER_OPERATOR(lookup_table_grad, ops::LookupTableOpGrad,
                  ops::LookupTableOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(lookup_table, ops::LookupTableKernel<float>,
                       ops::LookupTableKernel<double>);
REGISTER_OP_CPU_KERNEL(lookup_table_grad, ops::LookupTableGradKernel<float>,
                       ops::LookupTableGradKernel<double>);
