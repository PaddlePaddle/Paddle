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

#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/detail/safe_ref.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {

class ReorderLoDTensorByRankTableOpProtoMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  ReorderLoDTensorByRankTableOpProtoMaker(OpProto *proto,
                                          OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor) the input lod tensor need to be reordered.");
    AddInput("RankTable",
             "(LoDRankTable) the rank table that input need follow");
    AddOutput("Out", "(LoDTensor) reordered lod tensor");
    AddComment(R"DOC(ReorderLoDTensorByRankTable

Reorder the input X by the rank of `RankTable`. If `RankTable` is ordered by
index [3, 0, 2, 1]. Input X will reorder its sequence, the third sequence of
X will be the first sequence of Output.

NOTE: The RankTable does not need to be calculated by X.

For example:
The X = [Seq0, Seq1, Seq2, Seq3]. The indices of RankTable are [3, 0, 2, 1].

The Out =  [Seq3, Seq0, Seq2, Seq1] with correct LoD information.
)DOC");
  }
};

class ReorderLoDTensorByRankTableBase : public framework::OperatorBase {
 public:
  ReorderLoDTensorByRankTableBase(const std::string &type,
                                  const framework::VariableNameMap &inputs,
                                  const framework::VariableNameMap &outputs,
                                  const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::Place &place) const override {
    auto &x =
        detail::Ref(scope.FindVar(Input("X")),
                    "Cannot find input lod tensor variable %s", Input("X"))
            .Get<framework::LoDTensor>();
    auto &rank_table = detail::Ref(scope.FindVar(Input("RankTable")),
                                   "Cannot find input rank table variable %s",
                                   Input("RankTable"))
                           .Get<framework::LoDRankTable>();
    auto &out =
        *detail::Ref(scope.FindVar(Output("Out")),
                     "Cannot find output lod tensor variable %s", Output("Out"))
             .GetMutable<framework::LoDTensor>();

    out.Resize(x.dims());
    out.mutable_data(x.place(), x.type());
    this->process(place, x, rank_table, &out);
  }

 protected:
  virtual void process(const platform::Place &place,
                       const framework::LoDTensor &x,
                       const framework::LoDRankTable &rank_table,
                       framework::LoDTensor *out) const = 0;

  struct AbsoluteRankTableItem {
    size_t offset;  // the absolute/accumulated offset.
    size_t length;  // the length
    framework::LoD lod;
  };

  std::vector<AbsoluteRankTableItem> GetAbsoluteOffsetAndLengthByLoDRankTable(
      const framework::LoDTensor &x) const {
    std::vector<AbsoluteRankTableItem> absolute_table;
    size_t level = 0;
    size_t size = x.lod()[level].size();

    for (size_t i = 0; i < size - 1; ++i) {
      auto lod_offset =
          framework::GetSubLoDAndAbsoluteOffset(x.lod(), i, i + 1, level);

      auto &offset = lod_offset.second;

      absolute_table.emplace_back();
      absolute_table.back().length = offset.second - offset.first;
      absolute_table.back().offset = offset.first;
      absolute_table.back().lod = lod_offset.first;
    }
    return absolute_table;
  }

  size_t CopyTensorAndLod(const platform::Place &place,
                          const AbsoluteRankTableItem &item,
                          const framework::LoDTensor &x,
                          framework::LoDTensor *out, size_t out_offset) const {
    auto &out_lod = *out->mutable_lod();
    auto len = item.length;
    auto x_offset = item.offset;

    if (out_lod.empty()) {
      for (size_t i = 0; i < item.lod.size(); ++i) {
        out_lod.push_back(std::vector<size_t>({0}));
      }
    }

    for (size_t i = 0; i < out_lod.size(); ++i) {
      auto &out_v = out_lod[i];
      auto &new_lod_v = item.lod[i];

      for (auto &detail : new_lod_v) {
        out_v.push_back(out_v.back() + detail);
      }
    }

    auto x_sliced = x.Slice(x_offset, x_offset + len);
    auto out_sliced = out->Slice(out_offset, out_offset + len);

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Get();
    auto &dev_ctx = *pool.Borrow(place);
    framework::CopyFrom(x_sliced, out_sliced.place(), dev_ctx, &out_sliced);
    out_offset += len;
    return out_offset;
  }
};

class ReorderLoDTensorByRankTableOp : public ReorderLoDTensorByRankTableBase {
 public:
  ReorderLoDTensorByRankTableOp(const std::string &type,
                                const framework::VariableNameMap &inputs,
                                const framework::VariableNameMap &outputs,
                                const framework::AttributeMap &attrs)
      : ReorderLoDTensorByRankTableBase(type, inputs, outputs, attrs) {}

 protected:
  void process(const platform::Place &place, const framework::LoDTensor &x,
               const framework::LoDRankTable &rank_table,
               framework::LoDTensor *out) const override {
    auto absolute_table = GetAbsoluteOffsetAndLengthByLoDRankTable(x);
    size_t out_offset = 0;
    out->mutable_lod()->clear();
    for (auto &item : rank_table.items()) {
      PADDLE_ENFORCE_LT(item.index, absolute_table.size());
      out_offset = CopyTensorAndLod(place, absolute_table[item.index], x, out,
                                    out_offset);
    }
  }
};

class IdentityInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    context->SetOutputDim("Out", context->GetInputDim("X"));
  }
};

class ReorderLodTensorByRankGradOpMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("reorder_lod_tensor_by_rank_grad");
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetInput("RankTable", Input("RankTable"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

class ReorderLoDTensorByRankGradOp : public ReorderLoDTensorByRankTableBase {
 public:
  ReorderLoDTensorByRankGradOp(const std::string &type,
                               const framework::VariableNameMap &inputs,
                               const framework::VariableNameMap &outputs,
                               const framework::AttributeMap &attrs)
      : ReorderLoDTensorByRankTableBase(type, inputs, outputs, attrs) {}

 protected:
  void process(const platform::Place &place, const framework::LoDTensor &x,
               const framework::LoDRankTable &rank_table,
               framework::LoDTensor *out) const override {
    auto absolute_table = GetAbsoluteOffsetAndLengthByLoDRankTable(x);

    // offsets = enumerate([item.index for item in rank_table.items()])
    std::vector<std::pair<size_t, size_t>> offsets;
    offsets.reserve(rank_table.items().size());
    for (size_t i = 0; i < rank_table.items().size(); ++i) {
      offsets.push_back({i, rank_table.items()[i].index});
    }

    // offsets.sort(key=lambda x: x[1])
    std::sort(
        offsets.begin(), offsets.end(),
        [](const std::pair<size_t, size_t> &a,
           const std::pair<size_t, size_t> &b) { return a.second < b.second; });

    // Copy TensorAndLod
    size_t out_offset = 0;
    for (auto &offset : offsets) {
      out_offset = this->CopyTensorAndLod(place, absolute_table[offset.first],
                                          x, out, out_offset);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(reorder_lod_tensor_by_rank,
                  ops::ReorderLoDTensorByRankTableOp,
                  ops::ReorderLodTensorByRankGradOpMaker,
                  ops::ReorderLoDTensorByRankTableOpProtoMaker,
                  ops::IdentityInferShape);
REGISTER_OPERATOR(reorder_lod_tensor_by_rank_grad,
                  ops::ReorderLoDTensorByRankGradOp, ops::IdentityInferShape);
