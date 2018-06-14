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
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

constexpr int64_t kNoPadding = -1;

class LookupSparseTableInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LookupSparseTableOp should not be null.");
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
    unsigned int seed = static_cast<unsigned int>(Attr<int>("seed"));
    float min = Attr<float>("min");
    float max = Attr<float>("max");
    bool auto_grown_table = Attr<bool>("auto_grown_table");

    PADDLE_ENFORCE(out_var->IsType<framework::LoDTensor>(),
                   "The type of Out var should be LodTensor.");
    PADDLE_ENFORCE(w_var->IsType<framework::SelectedRows>(),
                   "The type of W var should be SelectedRows.");
    PADDLE_ENFORCE(ids_var->IsType<framework::LoDTensor>(),
                   "The type of Ids var should be LoDTensor.");
    auto &ids_t = ids_var->Get<framework::LoDTensor>();
    auto out_t = out_var->GetMutable<framework::LoDTensor>();
    auto w_t = w_var->GetMutable<framework::SelectedRows>();
    std::vector<int64_t> keys;
    keys.resize(ids_t.numel());
    for (int64_t i = 0; i < ids_t.numel(); ++i) {
      keys[i] = ids_t.data<int64_t>()[i];
    }

    // TODO(Yancey1989): support CUDA Place for the sparse table
    platform::CPUPlace cpu;
    auto out_shape = w_t->value().dims();
    out_shape[0] = keys.size();
    out_t->Resize(out_shape);
    out_t->mutable_data(cpu, w_t->value().type());
    PADDLE_ENFORCE_EQ(framework::ToDataType(w_t->value().type()),
                      framework::proto::VarType::FP32,
                      "The sparse table only support FP32");
    auto non_keys_pair = w_t->Get(keys, out_t);
    if (!auto_grown_table) {
      PADDLE_ENFORCE_EQ(non_keys_pair.size(), static_cast<size_t>(0),
                        "there is some keys does exists in the sparse table.");
    }
    auto value_shape = w_t->value().dims();
    value_shape[0] = 1;
    for (const auto &it : non_keys_pair) {
      const auto key = it.first;
      const auto index = it.second;
      framework::Tensor value;
      value.Resize(value_shape);
      auto data = value.mutable_data<float>(cpu);

      std::minstd_rand engine;
      engine.seed(seed);
      std::uniform_real_distribution<float> dist(min, max);
      int64_t size = value.numel();
      for (int64_t i = 0; i < size; ++i) {
        data[i] = dist(engine);
      }
      w_t->Set(key, value);
      memory::Copy(cpu, out_t->mutable_data<float>(cpu) + index * value.numel(),
                   cpu, value.data<float>(), value.numel() * sizeof(float));
    }
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
    AddAttr<float>("min",
                   "(float, default -1.0) "
                   "Minimum value of uniform random")
        .SetDefault(-1.0f);
    AddAttr<float>("max",
                   "(float, default 1.0) "
                   "Maximum value of uniform random")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "(int, default 0) "
                 "Random seed used for generating samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time.")
        .SetDefault(0);
    AddAttr<bool>("auto_grown_table",
                  "(bool default false)"
                  "Whether create new value if for nonexistent key.")
        .SetDefault(true);
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
REGISTER_OPERATOR(lookup_sparse_table, ops::LookupSparseTableOp,
                  ops::LookupSparseTableInferShape,
                  ops::LookupSparseTableOpMaker,
                  paddle::framework::EmptyGradOpMaker);
