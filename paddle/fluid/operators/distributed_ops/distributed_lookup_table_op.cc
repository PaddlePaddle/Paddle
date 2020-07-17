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

class DistributedLookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
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

    for (auto& ids_dim : ids_dims) {
      PADDLE_ENFORCE_EQ(ids_dim.size(), 2,
                        "The dimension of the 'Ids' tensor must be 2.");
    }

    auto lookup_tables =
        ctx->Attrs().Get<std::vector<std::string>>("table_names");
    auto height_sections =
        ctx->Attrs().Get<std::vector<int64_t>>("height_sections");
    auto endpoints = ctx->Attrs().Get<std::vector<std::string>>("endpoints");
    auto lookup_table_version =
        ctx->Attrs().Get<std::string>("lookup_table_version");

    PADDLE_ENFORCE(lookup_tables.size() == height_sections.size() &&
                       lookup_tables.size() == endpoints.size() &&
                       lookup_tables.size() != 0,
                   "Attrs lookup_tables/height_sections/endpoints must have "
                   "save size and can not be 0.");

    auto outputs_dims = std::vector<framework::DDim>();

    for (auto& ids_dim : ids_dims) {
      if (lookup_table_version == "lookup_table") {
        outputs_dims.push_back(
            framework::make_ddim({ids_dim[0], table_dims[1]}));
      } else if (lookup_table_version == "lookup_table_v2") {
        outputs_dims.push_back(framework::make_ddim(
            {static_cast<int64_t>(ids_dim[0]), static_cast<int64_t>(ids_dim[1]),
             static_cast<int64_t>(table_dims[1])}));
      }
    }

    ctx->SetOutputsDim("Outputs", outputs_dims);
    ctx->ShareLoD("Ids", /*->*/ "Outputs");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::proto::VarType::Type(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

template <typename T>
class DistributedLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto ids_vars = context.MultiInputVar("Ids");
    auto emb_vars = context.MultiOutput<framework::Tensor>("Embeddings");

    auto id_names = context.InputNames("Ids");
    auto embedding_name = context.InputNames("W").front();
    auto out_names = context.OutputNames("Outputs");

    auto lookup_tables = context.Attr<std::vector<std::string>>("table_names");
    auto height_sections =
        context.Attr<std::vector<int64_t>>("height_sections");
    auto endpoints = context.Attr<std::vector<std::string>>("endpoints");
    auto lookup_table_version =
        context.Attr<std::string>("lookup_table_version");

    operators::distributed::prefetchs(
        id_names, out_names, embedding_name, false, lookup_tables, endpoints,
        height_sections, context, context.scope());

    if (lookup_table_version == "lookup_table_v2") {
      auto& scope = context.scope();
      auto emb_dim =
          scope.FindVar(embedding_name)->Get<framework::LoDTensor>().dims()[1];

      for (size_t i = 0; i < id_names.size(); ++i) {
        auto* id_var = scope.FindVar(id_names[i]);
        auto* out_var = scope.FindVar(out_names[i]);
        auto* id_tensor = id_var->GetMutable<framework::LoDTensor>();
        auto* out_tensor = out_var->GetMutable<framework::LoDTensor>();

        auto id_dims = id_tensor->dims();
        out_tensor->Resize(framework::make_ddim(
            {static_cast<int64_t>(id_dims[0]), static_cast<int64_t>(id_dims[1]),
             static_cast<int64_t>(emb_dim)}));
      }
    }
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

    AddAttr<std::string>(
        "lookup_table_version",
        "(string, default lookup_table) "
        "To distinguish between different versions of embedding OP")
        .SetDefault(std::string("lookup_table"));

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
                       ops::DistributedLookupTableKernel<float>);
