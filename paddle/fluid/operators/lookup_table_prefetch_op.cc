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

#include <algorithm>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/parameter_prefetch.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class LookupTablePrefetchInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input(Ids) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(W) of LookupTableOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Embeddings"),
                   "Output(Embeddings) of LookupTableOp should not be null.");

    auto ids_dims = ctx->GetInputsDim("Ids");
    auto table_dims = ctx->GetInputDim("W");

    PADDLE_ENFORCE_EQ(table_dims.size(), 2,
                      "Only 2 dimensions of the 'Embedding' is supported.");
    for (size_t i = 0; i < ids_dims.size(); ++i) {
      PADDLE_ENFORCE_EQ(ids_dims[i].size(), 2,
                        "The dimension of the 'Ids' tensor must be 2.");
      PADDLE_ENFORCE_EQ(ids_dims[i][1], 1,
                        "The last dimension of the 'Ids' tensor must be 1.");
    }

    auto lookup_tables =
        ctx->Attrs().Get<std::vector<std::string>>("lookup_tables");
    auto height_sections =
        ctx->Attrs().Get<std::vector<int64_t>>("height_sections");
    auto endpoints = ctx->Attrs().Get<std::vector<std::string>>("endpoints");

    PADDLE_ENFORCE(lookup_tables.size() == height_sections.size() &&
                       lookup_tables.size() == endpoints.size() &&
                       lookup_tables.size() != 0,
                   "Attrs lookup_tables/height_sections/endpoints must have "
                   "save size and can not be 0.");

    auto outputs_dims = std::vector<framework::DDim>();
    for (size_t i = 0; i < ids_dims.size(); ++i) {
      auto o_dims = framework::vectorize(
          framework::slice_ddim(ids_dims, 0, ids_rank - 1));
      o_dims.push_back(table_dims[1]);
      outputs_dims.push_back(o_dims);
    }

    ctx->SetOutputsDim("Embeddings", outputs_dims);
    ctx->ShareLoD("Ids", /*->*/ "Embeddings");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class LookupTablePrefetchOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto ids_vars = context.MultiInputVar("Ids");
    auto table_var = context.InputVar("W");
    auto emb_vars = ctx.MultiOutput<framework::Tensor>("Embeddings");

    auto lookup_tables =
        context.Attr<std::vector<std::string>>("lookup_tables");
    auto height_sections =
        context.Attr<std::vector<int64_t>>("height_sections");
    auto endpoints = context.Attr<std::vector<std::string>>("endpoints");

    operators::distributed::multi_prefetch(
        context.Inputs("Ids"), context.Outputs("Embeddings"), lookup_tables,
        epmap, height_sections, context, context.scope());
  }
};

class LookupTablePrefetchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LoDTensor) Ids's type should be LoDTensor"
             "THe ids to be looked up in W.")
        .AsDuplicable();

    AddInput("W",
             "(Tensor) The input represents embedding tensors, "
             "which is a learnable parameter.");

    AddOutput("Embeddings",
              "(LoDTensor) The lookup results, which have the same type as W.")
        .AsDuplicable();

    AddAttr<std::vector<std::string>>(
        "lookup_tables",
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
REGISTER_OPERATOR(lookup_table_prefetch, ops::LookupTablePrefetchOp,
                  ops::LookupTablePrefetchInferShape,
                  ops::LookupTablePrefetchOpMaker,
                  paddle::framework::EmptyGradOpMaker);
