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
#include "paddle/fluid/operators/distributed/large_scale_kv.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class LookupSparseTableGradSplitInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class LookupSparseTableGradSplitOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &in_grad =
        scope.FindVar(Input("Grad"))->Get<framework::SelectedRows>();

    auto in_rows = in_grad.rows();
    auto in_value = in_grad.value();

    auto *out_row =
        scope.FindVar(Output("Row"))->GetMutable<framework::LoDTensor>();
    out_row->Resize(
        framework::make_ddim({static_cast<int64_t>(in_rows.size()), 1}));

    auto *t = out_row->mutable_data<int64_t>(dev_place);
    std::memcpy(t, rows.data(), rows.size() * sizeof(int64_t));

    auto *out_value = scope.FindVar(Output("Value"));
    auto *out_t = out_value->GetMutable<framework::LoDTensor>();
    framework::TensorCopy(in_value, dev_place, out_t);
  }
};

class LookupSparseTableGradSplitOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Grad",
             "(SelectedRows) Ids's type should be SelectedRows"
             "THe ids to be looked up in W.");
    AddOutput("Row",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.");
    AddOutput("Value",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.");
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
    lookup_sparse_table_grad_split, ops::LookupSparseTableGradSplitOp,
    ops::LookupSparseTableGradSplitInferShape,
    ops::LookupSparseTableGradSplitOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
