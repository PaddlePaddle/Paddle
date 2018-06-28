/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lookup_table_op.h"
#include "paddle/fluid/framework/var_type_inference.h"

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

    auto ids_var_type = ctx->GetInputsVarType("Ids").front();
    // The type of Ids(Input) is SelectedRows or LoDTensor, when Ids's type
    // is LoDTensor, this tensor contains the ids to be looked up in W
    // and it must be a column vector with rank = 2 while the 2nd dimension
    // size must be 1, when Ids's type is SelectedRows, the rows of Ids
    // contains the ids to be looked up in W;
    if (ids_var_type == framework::proto::VarType::LOD_TENSOR) {
      PADDLE_ENFORCE_EQ(ids_dims.size(), 2);
      PADDLE_ENFORCE_EQ(ids_dims[1], 1);
    }

    ctx->SetOutputDim("Out", {ids_dims[0], table_dims[1]});
    ctx->ShareLoD("Ids", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");
    AddInput(
        "Ids",
        "(Tensor or SelectedRows) Ids's type can be Tensor or "
        "SelectedRows, when Ids's type is Tensor, this tensor contains "
        "the ids to be looked up in W and it must be a column vector with "
        "rank = 2 while the 2nd dimension size must be 1; when Ids's type is "
        "SelectedRows, the rows of Ids contains the ids to be looked up "
        "in W.");
    AddOutput("Out",
              "(Tensor or SelectedRows) The lookup results, which have the "
              "same type as W.");
    AddAttr<bool>("is_sparse",
                  "(boolean, default false) "
                  "Sparse update.")
        .SetDefault(false);
    AddAttr<bool>("is_distributed",
                  "(boolean, default false) distributed lookup table.")
        .SetDefault(false);
    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(kNoPadding);
    AddComment(R"DOC(
Lookup Table Operator.

This operator is used to perform lookups on the parameter W,
then concatenated into a dense or sparse tensor.

The type of Ids(Input) is SelectedRows, Tensor or LoDTensor, when Ids's
type is SelectedRows, the rows of Ids contains the ids to be looked up in W;
when Ids's type is Tensor, this tensor contains the ids to be looked up in W
and it must be a column vector with rank = 2 while the 2nd dimension size must be 1,
at this time, Ids can carry the LoD (Level of Details) information, or not, and
the output only shares the LoD information with input Ids.


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
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto out_var_name = op_desc.Output(framework::GradVarName("W")).front();
    auto attr = op_desc.GetAttr("is_sparse");
    bool is_sparse = boost::get<bool>(attr);
    if (is_sparse) {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to SelectedRows";
      block->Var(out_var_name)
          ->SetType(framework::proto::VarType::SELECTED_ROWS);
    } else {
      VLOG(3) << "lookup_table_grad op " << framework::GradVarName("W")
              << " is set to LoDTensor";
      block->Var(out_var_name)->SetType(framework::proto::VarType::LOD_TENSOR);
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
