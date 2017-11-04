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
#include <numeric>
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

static inline size_t CalcOffsetByLoD(const framework::LoD &lod, size_t idx) {
  for (const auto &level : lod) {
    idx = level.at(idx);
  }
  return idx;
}

class ArrayToLoDTensorOp : public framework::OperatorBase {
 public:
  ArrayToLoDTensorOp(const std::string &type,
                     const framework::VariableNameMap &inputs,
                     const framework::VariableNameMap &outputs,
                     const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensorArray>();
    auto &rank_table =
        scope.FindVar(Input("RankTable"))->Get<framework::LoDRankTable>();
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();

    // Check dims of input's elements and infer output's dim
    PADDLE_ENFORCE(!x.empty(), "There's no element in the input array.");
    int rank = x[0].dims().size();
    framework::DDim ins_dims = framework::slice_ddim(x[0].dims(), 1, rank);
    int64_t batch_size = 0;
    for (size_t i = 1; i < x.size(); ++i) {
      PADDLE_ENFORCE_EQ(framework::slice_ddim(x[i].dims(), 1, rank), ins_dims,
                        "The dimension of the %zu'th element in LoDTensorArray "
                        "differs from previous ones.",
                        i);
      batch_size += x[i].dims()[0];
    }
    framework::DDim out_dims = framework::make_ddim(
        framework::vectorize(ins_dims).push_front(batch_size));
    out->Resize(out_dims);

    auto &table_items = rank_table.items();
    std::vector<size_t> table_item_idx(rank_table_items.size());
    std::iota(std::begin(table_item_idx), std::end(table_item_idx), 0);
    std::sort(table_item_idx.begin(), table_item_idx.end(),
              [&](const size_t &a, const size_t &b) {
                return table_items[a].index < table_items[b].index;
              });

    // Copy data from input x to output out
    for (const size_t &idx : table_item_idx) {
      size_t seq_len = table_items[idx].length;
      for (size_t x_idx = 0; x_idx < seq_len; ++x_idx) {
      }
    }
  }
};

class ArrayToLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ArrayToLoDTensorOpProtoMaker(framework::OpProto *proto,
                               framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(std::vector<LodTensor>) A vector of tensors that is going to "
             "be casted to a big LoDTensor.");
    AddInput("RankTable",
             "(LoDRankTable) RankTable provides the coarse lod infomation to "
             "build the output LoDTensor. See "
             "'paddle/framework/lod_rank_table.h' for more details.");
    AddOutput("Out", "(LoDTensor) The LoDTensor formed by input tensor array.");
    AddComment(
        R"DOC(This Op build a big LoDTensor from a std::vector<LoDTensor> 
          and a LoDRankTable. It is supposed to be used in getting dynamic RNN's
          outputs back to a normal LoDTensor. The std::vector<LoDTensor> 
          would be the output of RNN Op and the LoDRankTable would be build 
          with RNN's input.)DOC");
  }
};

class ArrayToLoDTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("X"),
                   "ArrayToLoDTensorOp must has input X.");
    PADDLE_ENFORCE(context->HasInput("RankTable"),
                   "ArrayToLoDTensorOp must has input RankTable.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(array_to_lod_tensor, ops::ArrayToLoDTensorOp,
                  ops::ArrayToLoDTensorOpProtoMaker,
                  ops::ArrayToLoDTensorInferShape);
