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

#include "paddle/fluid/framework/details/memory_optimize_pass.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>
#include "glog/logging.h"

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {
namespace details {

static const std::unordered_set<std::string> kSUB_BLOCK_OPS = {
    "while",
    "while_grad",
    "parallel_do",
    "parallel_do_grad",
    "conditional_block",
    "conditional_block_grad"};

bool MemoryOptimizePass::IsValidVar(ir::Node* node) const {
  PADDLE_ENFORCE(node->IsVar(), "Expect Variable");
  // TODO(dzhwinter): ir node maybe empty
  // if (node->Var() == nullptr || node->Op() == nullptr) {
  //   return false;
  // }
  VarDesc* desc = node->Var();
  // only LoDTensor can be reused
  if (desc->Name() == "@EMPTY@" || desc->Persistable() ||
      desc->GetType() != proto::VarType::LOD_TENSOR ||
      desc->GetShape().size() == 0) {
    return false;
  }
  // TODO(dzhwinter): force_cpu var can not be reused yet.
  // can get the runtime place from executor.
  for (auto& generated_op : node->inputs) {
    if (generated_op->Name() == "fill_constant" &&
        generated_op->Op()->HasAttr("force_cpu")) {
      skip_set_.insert(node);
    }
  }
  if (skip_set_.find(node) != skip_set_.end()) {
    return false;
  }
  return true;
}

ir::Node* MemoryOptimizePass::SearchMatch(ir::Node* var) const {
  // TODO(dzhwinter): dynamic plan matching, datatype convert matching.
  auto cmp_var = [&](ir::Node* lhs, ir::Node* rhs) -> bool {
    std::vector<int64_t> sa = lhs->Var()->GetShape();
    std::vector<int64_t> sb = rhs->Var()->GetShape();
    if (sa[0] == -1 || sb[0] == -1) {
      if (sa[0] != sb[0]) return false;
    }
    return std::abs(std::accumulate(sa.begin(), sa.end(), 1)) ==
               std::abs(std::accumulate(sb.begin(), sb.end(), 1)) &&
           lhs->Var()->GetDataType() == rhs->Var()->GetDataType();
  };
  for (auto& cache_var : pool_) {
    if (cmp_var(cache_var, var)) {
      return cache_var;
    }
  }
  return nullptr;
}

const std::string MemoryOptimizePass::DebugString(ir::Node* var) const {
  std::stringstream ss;
  ss << var->Name();
  ss << "[";
  auto shape = var->Var()->GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != shape.size() - 1) {
      ss << shape[i] << ",";
    } else {
      ss << shape[i];
    }
  }
  ss << "]";
  return ss.str();
}

// template<typename Container>
std::string ToString(const std::set<ir::Node*>& pool) {
  std::stringstream ss;
  for (auto& var : pool) {
    ss << var->Name() << ",";
  }
  return ss.str();
}

std::string ToString(const std::unordered_set<ir::Node*>& pool) {
  std::stringstream ss;
  for (auto& var : pool) {
    ss << var->Name() << ",";
  }
  return ss.str();
}

std::unique_ptr<ir::Graph> MemoryOptimizePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  cfg_.reset(new ControlFlowGraph(*graph));
  cfg_->DataAnalysis();

  for (size_t i = 0; i < cfg_->Ops().size(); ++i) {
    auto op = cfg_->Ops()[i];
    VLOG(3) << op->Name() << " " << i << " live_in "
            << ToString(cfg_->LiveIn(op));
    VLOG(3) << op->Name() << " " << i << " live_out "
            << ToString(cfg_->LiveOut(op));
  }

  // for (auto& op : cfg_->Ops()) {
  for (size_t i = 0; i < cfg_->Ops().size(); ++i) {
    auto op = cfg_->Ops()[i];
    if (kSUB_BLOCK_OPS.find(op->Name()) != kSUB_BLOCK_OPS.end()) {
      continue;
    }
    // 1. find unused vars, fill pool
    for (auto& var : cfg_->LiveIn(op)) {
      if (cfg_->LiveOut(op).find(var) == cfg_->LiveOut(op).end()) {
        if (IsValidVar(var)) {
          pool_.insert(var);
        }
      }
    }
    VLOG(3) << op->Name() << " " << i << " " << ToString(pool_);
    // 2. reuse var matching
    for (auto& output_var : cfg_->Def(op)) {
      if (IsValidVar(output_var)) {
        auto* cache_var = SearchMatch(output_var);
        if (cache_var != nullptr) {
          auto index = static_cast<int>(
              std::distance(pool_.find(cache_var), pool_.begin()));
          VLOG(3) << string::Sprintf(
              "Hit Cache !!! cache pool index %d, var is %s, cached var %s",
              index, DebugString(output_var), DebugString(cache_var));
          cfg_->UpdateGraph(output_var, cache_var, index);
          graph->RemoveNode(output_var);
          pool_.erase(cache_var);
        }
      }
    }
    VLOG(3) << "after " << op->Name() << " " << i << " " << ToString(pool_);
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(memory_optimize_pass,
              paddle::framework::details::MemoryOptimizePass);
