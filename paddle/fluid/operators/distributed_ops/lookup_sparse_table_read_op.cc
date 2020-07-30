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

class LookupSparseTableReadInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class LookupSparseTableReadOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto init = Attr<bool>("init");

    auto &id_tensor = scope.FindVar(Input("Ids"))->Get<framework::LoDTensor>();
    auto *id_data = id_tensor.data<int64_t>();
    auto tablename = Attr<std::string>("tablename");
    auto value_names = Attr<std::vector<std::string>>("value_names");
    auto out_names = Outputs("Out");

    std::vector<int64_t> ids;
    for (int64_t i = 0; i < id_tensor.numel(); ++i) {
      ids.push_back(id_data[i]);
    }

    std::vector<std::vector<std::vector<float> *>> values;
    std::vector<int64_t> dims;

    auto *ins = distributed::LargeScaleKV::GetInstance();

    if (init) {
      ins->Get(tablename)->Init(ids);
      ins->Get(tablename)->Get(ids, value_names, &values);
    } else {
      ins->Get(tablename)->Get(ids, value_names, &values);
    }

    ins->Get(tablename)->Dims(value_names, &dims);

    platform::CPUPlace cpu;
    std::vector<float *> tensors;

    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      auto out_var = scope.FindVar(out_names[i]);
      auto out_t = out_var->GetMutable<framework::LoDTensor>();

      std::vector<int64_t> o_dims;
      o_dims.push_back(static_cast<int64_t>(ids.size()));
      o_dims.push_back(dims[i]);
      out_t->Resize(framework::make_ddim(o_dims));
      auto *out_d = out_t->mutable_data<float>(cpu);
      tensors.push_back(out_d);
    }

    for (int i = 0; i < static_cast<int>(values.size()); i++) {
      for (int j = 0; j < static_cast<int>(tensors.size()); j++) {
        std::memcpy(tensors[j] + i * dims[j], values[i][j]->data(),
                    sizeof(float) * dims[j]);
      }
    }
  }
};

class LookupSparseTableReadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LoDTensor) Ids's type should be LoDTensor"
             "THe ids to be looked up in W.");
    AddOutput("Out",
              "(LoDTensor) The lookup results, which have the "
              "same type as W.")
        .AsDuplicable();

    AddAttr<std::string>("tablename",
                         "(string)"
                         "sparse table name");

    AddAttr<std::vector<std::string>>("value_names",
                                      "(strings)"
                                      "sparse table name");

    AddAttr<bool>("init", " for test init large scale kv").SetDefault(false);

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
    lookup_sparse_table_read, ops::LookupSparseTableReadOp,
    ops::LookupSparseTableReadInferShape, ops::LookupSparseTableReadOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
