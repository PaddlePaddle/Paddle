// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/details/fuse_all_reduce_graph_builder.h"
#include <list>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"

namespace paddle {
namespace framework {
namespace details {
std::unique_ptr<SSAGraph> FuseAllReduceGraphBuilder::Build(
    const ProgramDesc &program) const {
  // TODO(yy): Complete this method.
  auto graph = builder_->Build(program);

  auto all_reduce_ops = GetNotDependedAllReduceOp(graph.get());

  for (auto &op_group : all_reduce_ops) {
    FuseAllReduceOp(graph.get(), std::move(op_group));
  }
  return graph;
}

inline static bool IsParentOpInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  if (cur_ops.count(offset_map.at(op))) {
    return true;
  }

  for (auto *in : op->Inputs()) {
    if (in->generated_op_ &&
        IsParentOpInCurrentOp(in->generated_op_, cur_ops, offset_map)) {
      return true;
    }
  }

  return false;
}

inline static bool IsChildOpInCurrentOp(
    OpHandleBase *op, const std::unordered_set<size_t> &cur_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map) {
  if (cur_ops.count(offset_map.at(op))) {
    return true;
  }

  for (auto *out : op->Outputs()) {
    for (auto *out_op : out->pending_ops_) {
      if (IsChildOpInCurrentOp(out_op, cur_ops, offset_map)) {
        return true;
      }
    }
  }

  return false;
}

inline static void ResolveOpDeps(
    const std::vector<std::unique_ptr<OpHandleBase>> &all_ops,
    const std::unordered_map<OpHandleBase *, size_t> &offset_map,
    std::list<std::unordered_set<size_t>>::iterator cur_it,
    std::list<std::unordered_set<size_t>> *res) {
  std::unordered_set<size_t> before_deps;
  std::unordered_set<size_t> after_deps;
  std::unordered_set<size_t> cur_ops;

  for (size_t pos : *cur_it) {
    if (cur_ops.empty()) {
      cur_ops.emplace(pos);
      continue;
    }
    auto *op_handle = all_ops[pos].get();

    if (IsParentOpInCurrentOp(op_handle, cur_ops, offset_map)) {
      after_deps.emplace(pos);
    } else if (IsChildOpInCurrentOp(op_handle, cur_ops, offset_map)) {
      before_deps.emplace(pos);
    } else {
      cur_ops.emplace(pos);
    }
  }

  if (!before_deps.empty()) {
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, before_deps), res);
  }

  cur_it->swap(cur_ops);

  if (!after_deps.empty()) {
    ++cur_it;
    ResolveOpDeps(all_ops, offset_map, res->insert(cur_it, after_deps), res);
  }
}

std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>>
FuseAllReduceGraphBuilder::GetNotDependedAllReduceOp(SSAGraph *graph) const {
  std::vector<std::unique_ptr<OpHandleBase>> all_reduce_ops;

  for (size_t i = 0; i < graph->ops_.size();) {
    if (dynamic_cast<NCCLAllReduceOpHandle *>(graph->ops_[i].get())) {
      all_reduce_ops.emplace_back(graph->ExtractOp(i));
    } else {
      ++i;
    }
  }
  std::unordered_map<OpHandleBase *, size_t> offsets;
  std::list<std::unordered_set<size_t>> res;
  res.emplace_back();
  for (size_t i = 0; i < all_reduce_ops.size(); ++i) {
    offsets.emplace(all_reduce_ops[i].get(), i);
    res.back().emplace(i);
  }

  ResolveOpDeps(all_reduce_ops, offsets, res.begin(), &res);

  std::vector<std::unordered_set<std::unique_ptr<OpHandleBase>>> res_vec;
  for (auto &set : res) {
    res_vec.emplace_back();
    auto &pointer_set = res_vec.back();

    for (auto pos : set) {
      pointer_set.emplace(std::move(all_reduce_ops[pos]));
    }
  }

  return res_vec;
}
void FuseAllReduceGraphBuilder::FuseAllReduceOp(
    SSAGraph *graph,
    std::unordered_set<std::unique_ptr<OpHandleBase>> &&ops) const {}
}  // namespace details
}  // namespace framework
}  // namespace paddle
