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
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class LookupSparseTableInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "LookupSparseTable");
    auto shape_w = ctx->GetInputDim("W");
    auto shape_ids = ctx->GetInputDim("Ids");
    shape_w[0] = shape_ids.size();
    ctx->SetOutputDim("Out", shape_w);
  }
};

class LookupSparseTableOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto out_var = scope.FindVar(Output("Out"));
    auto w_var = scope.FindVar(Input("W"));
    auto ids_var = scope.FindVar(Input("Ids"));
    auto is_test = Attr<bool>("is_test");

    PADDLE_ENFORCE_EQ(out_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The type of Out var should be LodTensor."));
    PADDLE_ENFORCE_EQ(w_var->IsType<framework::SelectedRows>(), true,
                      platform::errors::InvalidArgument(
                          "The type of W var should be SelectedRows."));
    PADDLE_ENFORCE_EQ(ids_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The type of Ids var should be LoDTensor."));
    auto &ids_t = ids_var->Get<framework::LoDTensor>();
    auto out_t = out_var->GetMutable<framework::LoDTensor>();
    auto w_t = w_var->GetMutable<framework::SelectedRows>();

    // TODO(Yancey1989): support CUDA Place for the sparse table
    platform::CPUPlace cpu;
    auto out_shape = w_t->value().dims();
    out_shape[0] = ids_t.numel();
    out_t->Resize(out_shape);
    out_t->mutable_data(cpu, w_t->value().type());
    PADDLE_ENFORCE_EQ(w_t->value().type(), framework::proto::VarType::FP32,
                      platform::errors::InvalidArgument(
                          "The sparse table only support FP32"));
    w_t->Get(ids_t, out_t, true, is_test);
    out_t->set_lod(ids_t.lod());
  }
};

class LookupSparseTableOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("W",
             "(SelectedRows) The input represents embedding table, "
             "which is a learnable parameter.");
    AddInput("Ids",
             "(LoDTensor) Ids's type should be LoDTensor"
             "THe ids to be looked up in W.");
    AddOutput("Out",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.");
    AddAttr<int64_t>("padding_idx",
                     "(int64, default -1) "
                     "If the value is -1, it makes no effect to lookup. "
                     "Otherwise the given value indicates padding the output "
                     "with zeros whenever lookup encounters it in Ids.")
        .SetDefault(kNoPadding);
    AddAttr<bool>("auto_grown_table",
                  "(bool default false)"
                  "Whether create new value if for nonexistent key.")
        .SetDefault(true);
    AddAttr<bool>("is_test",
                  "In test mode, lookup_sparse_table will "
                  "return a 0 for unknown id")
        .SetDefault(false);
    AddComment(R"DOC(
Lookup Sprase Tablel Operator.

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
REGISTER_OPERATOR(
    lookup_sparse_table, ops::LookupSparseTableOp,
    ops::LookupSparseTableInferShape, ops::LookupSparseTableOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
