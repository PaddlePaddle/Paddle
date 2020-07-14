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

// examples: embedding:Param,Moment1,Moment2:64,64,64:0
constexpr char kLargeScaleKV[] = "large_scale_metas";
constexpr int64_t kNoPadding = -1;

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

class LookupSparseTableInitInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

void InitLargeScaleKV(std::vector<std::string> kv_attrs) {
  std::vector<distributed::SparseMeta> metas;

  for (auto attrs : kv_attrs) {
    std::vector<std::string> pieces;
    split(attrs, ':', &pieces);
    PADDLE_ENFORCE_EQ(
        pieces.size(), 8,
        platform::errors::InvalidArgument(
            "param, names, dims, mode, grad, cached_var, init_attrs"));

    std::string name;
    std::string grad_name;
    std::vector<std::string> value_names;
    std::vector<int> value_dims;
    distributed::Mode mode;
    std::vector<std::string> cached_names;
    std::vector<std::string> init_attrs;
    std::string entry_attr;

    name = pieces[0];
    split(pieces[1], ',', &value_names);

    std::vector<std::string> value_dims_str;
    split(pieces[2], ',', &value_dims_str);
    for (auto &str : value_dims_str) {
      value_dims.push_back(std::stoi(str));
    }

    mode = pieces[3] == "0" ? distributed::Mode::training
                            : distributed::Mode::infer;

    grad_name = pieces[4];
    split(pieces[5], ',', &cached_names);
    split(pieces[6], ',', &init_attrs);
    entry_attr = pieces[7];

    auto meta = distributed::SparseMeta();
    meta.name = name;
    meta.value_names = value_names;
    meta.value_dims = value_dims;
    meta.mode = mode;
    meta.grad_name = grad_name;
    meta.cached_varnames = cached_names;
    meta.initializer_attrs = init_attrs;
    meta.entry = entry_attr;

    VLOG(3) << "add sparse meta: " << meta.ToString();
    metas.push_back(meta);
  }

  distributed::LargeScaleKV::Init(metas);
  VLOG(3) << "init large scale kv with " << metas.size() << " params";
}

class LookupSparseTableInitOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto kv_attrs = Attr<std::vector<std::string>>(kLargeScaleKV);
    InitLargeScaleKV(kv_attrs);
  }
};

class LookupSparseTableInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddAttr<std::vector<std::string>>(kLargeScaleKV,
                                      "(string)"
                                      "sparse table name");
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
    lookup_sparse_table_init, ops::LookupSparseTableInitOp,
    ops::LookupSparseTableInitInferShape, ops::LookupSparseTableInitOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
