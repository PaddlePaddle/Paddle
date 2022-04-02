/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lookup_table_dequant_op.h"

#include <memory>

#include "paddle/fluid/framework/no_need_buffer_vars_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace operators {

class LookupTableDequantOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("W"), true,
        platform::errors::InvalidArgument(
            "Input(W) of LookupTableDequantOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Ids"), true,
        platform::errors::InvalidArgument(
            "Input(Ids) of LookupTableDequantOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Out"), true,
        platform::errors::InvalidArgument(
            "Output(Out) of LookupTableDequantOp should not be null."));

    auto table_dims = ctx->GetInputDim("W");
    auto ids_dims = ctx->GetInputDim("Ids");
    int ids_rank = ids_dims.size();
    VLOG(5) << "ids rank is " << ids_rank << std::endl;
    PADDLE_ENFORCE_EQ(
        table_dims.size(), 2,
        platform::errors::InvalidArgument(
            "ShapeError: The dimensions of the 'lookup table' must be 2. "
            "But received lookup table's dimensions = %d, "
            "lookup table's shape = [%s].",
            table_dims.size(), table_dims));
    PADDLE_ENFORCE_EQ(
        ids_dims[ids_rank - 1], 1,
        platform::errors::InvalidArgument(
            "ShapeError: The last dimensions of the 'Ids' tensor must be 1. "
            "But received Ids's last dimensions = %d, Ids's shape = [%s].",
            ids_dims[ids_rank - 1], ids_dims));

    auto output_dims =
        phi::vectorize(phi::slice_ddim(ids_dims, 0, ids_rank - 1));
    PADDLE_ENFORCE_GE(table_dims[1], 2,
                      platform::errors::InvalidArgument(
                          "the second dim of table_dims should be "
                          "greater or equal to 2, but the actual shape "
                          "is [%s]",
                          table_dims));

    output_dims.push_back((table_dims[1] - 2) * 4);
    ctx->SetOutputDim("Out", phi::make_ddim(output_dims));

    if (ctx->GetOutputsVarType("Out")[0] ==
        framework::proto::VarType::LOD_TENSOR) {
      ctx->ShareLoD("Ids", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "W");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTableDequantOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "This tensor is a quantized tensor");
    AddInput("Ids",
             "An input with type int64 "
             "contains the ids to be looked up in W. "
             "The last dimension size must be 1.");
    AddOutput("Out", "The lookup results, which have the same type as W.");
    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(kNoPadding);
    AddComment(R"DOC(
Lookup Table Dequant Operator.

The `W` input is a quantized parameter for the sake of saving memories.
This operator first index embeddings with `Ids`,
then dequantizes them and contact them as output (`Out`). 

The input Ids can carry the LoD (Level of Details) information,
or not. And the output only shares the LoD information with input Ids.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    lookup_table_dequant, ops::LookupTableDequantOp,
    ops::LookupTableDequantOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(lookup_table_dequant,
                       ops::LookupTableDequantKernel<float>);
