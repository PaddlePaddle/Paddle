/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/rank_sort_lod_op.h"

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {

/*
 * DyBatchSeqPosition stores indices of the basic element in tensor. It is used
 * after lod-tensor's re-assembling, its info can be used to recover the order
 * in original lod-tensor.
 */
struct DySeqMeta {
  DySeqMeta(size_t begin, size_t end, size_t ori_idx)
      : begin(begin), end(end), ori_idx(ori_idx) {}

  size_t begin;
  size_t end;  // not included
  size_t ori_idx;
};

std::vector<DySeqMeta> BuildLengthSortedMeta(const framework::LoDTensor& source,
                                             size_t level) {
  std::vector<DySeqMeta> meta;
  // collect meta for each sequence in some level
  auto lod = framework::SliceLevels(source.lod(), level, level + 1)[0];

  for (size_t seq_id = 0; seq_id < lod.size() - 1; seq_id++) {
    DySeqMeta seq_meta({lod[seq_id], lod[seq_id + 1], seq_id});
    meta.push_back(seq_meta);
  }

  PADDLE_ENFORCE_GT(meta.size(), 0, "meta is empty");

  // sort by length
  sort(meta.begin(), meta.end(), [](const DySeqMeta& a, const DySeqMeta& b) {
    return (a.end - a.begin) > (b.end - b.begin);
  });
  return meta;
}

void RankSortLoDOp::Run(const framework::Scope& scope,
                        const platform::DeviceContext& dev_ctx) const {
  framework::Variable* x = scope.FindVar(Input("X"));
  PADDLE_ENFORCE_NOT_NULL(x, "X is not set");
  const auto& tensor = x->Get<framework::LoDTensor>();
  auto lod_level = Attr<int>("lod_level");
  auto meta = BuildLengthSortedMeta(tensor, lod_level);
  // copy meta to tmp
  framework::LoDTensor tmp;
  tmp.Resize(
      framework::make_ddim(std::vector<int64_t>({(int64_t)meta.size(), 3})));
  auto* data = tmp.mutable_data<int64_t>(platform::CPUPlace());
  for (const auto& rcd : meta) {
    *(data++) = rcd.begin;
    *(data++) = rcd.end;
    *(data++) = rcd.ori_idx;
  }
  // copy cpu lodtensor to gpu
  auto* out_var = scope.FindVar(Output("Out"));
  PADDLE_ENFORCE_NOT_NULL(
      out_var, "Out variable [%s] should be created by framework first.",
      Output("Out"));
  auto* out_tensor = out_var->GetMutable<framework::LoDTensor>();
  out_tensor->Resize(tmp.dims());
  out_tensor->mutable_data(dev_ctx.GetPlace());
  out_tensor->CopyFrom(tmp, dev_ctx.GetPlace(), dev_ctx);
}

class RankSortLoDProtoAndCheckerMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  RankSortLoDProtoAndCheckerMaker(framework::OpProto* proto,
                                  framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "a LoDTensor");
    AddOutput("Out",
              "a Nx3 tensor, the format of each instance is [beginOffset, "
              "endOffset, seqId]");
    AddAttr<int>("lod_level", "the level of the LoDTensor to rank");
    AddComment(R"DOC(
Rank sort a LoDTensor's lod by sequences' lengths in descending order.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(
    rank_sort_lod, paddle::operators::RankSortLoDOp,
    paddle::operators::RankSortLoDProtoAndCheckerMaker);
