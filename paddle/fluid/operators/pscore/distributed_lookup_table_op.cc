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
#include "paddle/fluid/operators/pscore/distributed_lookup_table_op.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class DistributedLookupTableOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("Ids"), true,
                      platform::errors::InvalidArgument(
                          "Input(Ids) of LookupTableOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      platform::errors::InvalidArgument(
                          "Input(W) of LookupTableOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutputs("Outputs"), true,
                      platform::errors::InvalidArgument(
                          "Output(Outs) of LookupTableOp should not be null."));

    auto ids_dims = ctx->GetInputsDim("Ids");
    auto table_dims = ctx->GetInputDim("W");

    PADDLE_ENFORCE_EQ(
        table_dims.size(), 2,
        platform::errors::InvalidArgument(
            "Only 2 dimensions of the 'Embedding' is supported."));

    for (auto &ids_dim : ids_dims) {
      PADDLE_ENFORCE_EQ(ids_dim.size(), 2,
                        platform::errors::InvalidArgument(
                            "The dimension of the 'Ids' tensor must be 2."));
    }

    // for fluid.embedding
    auto lookup_table_version =
        ctx->Attrs().Get<std::string>("lookup_table_version");

    auto outputs_dims = std::vector<framework::DDim>();

    for (auto &ids_dim : ids_dims) {
      if (lookup_table_version == "lookup_table") {
        outputs_dims.push_back(phi::make_ddim({ids_dim[0], table_dims[1]}));
      } else if (lookup_table_version == "lookup_table_v2") {
        outputs_dims.push_back(phi::make_ddim(
            {static_cast<int64_t>(ids_dim[0]), static_cast<int64_t>(ids_dim[1]),
             static_cast<int64_t>(table_dims[1])}));
      }
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

    AddAttr<int>("table_id", "sparse table id").SetDefault(0);

    AddAttr<bool>("is_distributed",
                  "(boolean, default false) distributed lookup table.")
        .SetDefault(false);

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
        .SetDefault(kNoPadding);
    AddAttr<int>("dtype",
                 "(int, default 5 (FP32)) "
                 "Output data type")
        .SetDefault(framework::proto::VarType::FP32);

    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);

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
