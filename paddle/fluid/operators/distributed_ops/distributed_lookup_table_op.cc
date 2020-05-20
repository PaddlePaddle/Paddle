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

#include "paddle/fluid/operators/distributed_ops/distributed_lookup_table_op.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class DistributedLookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInputs("Ids"),
                   "Input(Ids) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(W) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutputs("Outputs"),
                   "Output(Outs) of LookupTableOp should not be null.");

    auto ids_dims = ctx->GetInputsDim("Ids");
    auto table_dims = ctx->GetInputDim("W");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2,
                      "Only 2 dimensions of the 'Embedding' is supported.");

    for (auto &ids_dim : ids_dims) {
      PADDLE_ENFORCE_EQ(ids_dim.size(), 2,
                        "The dimension of the 'Ids' tensor must be 2.");
      PADDLE_ENFORCE_EQ(ids_dim[1], 1,
                        "The last dimension of the 'Ids' tensor must be 1.");
    }

    auto lookup_tables =
        ctx->Attrs().Get<std::vector<std::string>>("table_names");
    auto height_sections =
        ctx->Attrs().Get<std::vector<int64_t>>("height_sections");
    auto endpoints = ctx->Attrs().Get<std::vector<std::string>>("endpoints");

    PADDLE_ENFORCE(lookup_tables.size() == height_sections.size() &&
                       lookup_tables.size() == endpoints.size() &&
                       lookup_tables.size() != 0,
                   "Attrs lookup_tables/height_sections/endpoints must have "
                   "save size and can not be 0.");

    auto outputs_dims = std::vector<framework::DDim>();

    for (auto &ids_dim : ids_dims) {
      outputs_dims.push_back(framework::make_ddim({ids_dim[0], table_dims[1]}));
    }

    ctx->SetOutputsDim("Outputs", outputs_dims);
    ctx->ShareLoD("Ids", /*->*/ "Outputs");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class DistributedLookupTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LoDTensor) Ids's type should be LoDTensor"
             "THe ids to be looked up in W.")
        .AsDuplicable();

    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");

    AddOutput("Outputs",
              "(LoDTensor) The lookup results, which have the same type as W.")
        .AsDuplicable();

    AddAttr<std::vector<std::string>>(
        "table_names",
        "(string vector, such as emb_block0, emb_block1)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({""});

    AddAttr<std::vector<int64_t>>("height_sections",
                                  "Height for each output SelectedRows.")
        .SetDefault(std::vector<int64_t>({}));

    AddAttr<std::vector<std::string>>(
        "endpoints",
        "(string vector, default 127.0.0.1:6164)"
        "Server endpoints in the order of input variables for mapping")
        .SetDefault({"127.0.0.1:6164"});

    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);

    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(distributed::kNoPadding);
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);

    AddComment(R"DOC(
Lookup Tablel Prefetch Operator.

This operator is used to perform lookup on parameter W,
then concatenated into a sparse tensor.

The type of Ids(Input) is SelectedRows, the rows of Ids contains
the ids to be looked up in W;
if the Id is not in the sparse table, this operator will return a
random value and set the value into the table for the next looking up.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(distributed_lookup_table, ops::DistributedLookupTableOp,
                  ops::DistributedLookupTableOpMaker);

REGISTER_OP_CPU_KERNEL(distributed_lookup_table,
                       ops::DistributedLookupTableKernel<
                           paddle::platform::CPUDeviceContext, float>);
