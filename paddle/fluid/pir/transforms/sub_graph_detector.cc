// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/transforms/sub_graph_detector.h"

#include <memory>

#include <iterator>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#ifdef PADDLE_WITH_CINN
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/cinn/utils/string.h"
#endif

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builder.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

#include "paddle/common/flags.h"

#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#endif
namespace pir {

std::vector<pir::Operation*> InverselyTopologicalSort(pir::Block* block) {
  std::vector<pir::Operation*> sort_ops;
  std::unordered_map<pir::Operation*, int> pending_count;
  // step 1: initialize pending_cout for defined op
  for (auto& op : *block) {
    if (pending_count.find(&op) == pending_count.end()) {
      pending_count[&op] = 0;
    }
    for (auto operand : GetUsedExternalValue(op)) {
      if (!operand || !operand.defining_op()) {
        continue;
      }
      auto* defined_op = operand.defining_op();
      if (pending_count.find(defined_op) != pending_count.end()) {
        ++pending_count[defined_op];
      } else {
        pending_count[defined_op] = 1;
      }
    }
  }

  std::queue<pir::Operation*> queue;
  for (auto& op : *block) {
    VLOG(4) << op.name() << " pending_count: " << pending_count[&op];
    if (pending_count[&op] == 0) {
      queue.push(&op);
    }
  }

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    VLOG(4) << "Pop Op: " << op->name();
    sort_ops.push_back(op);
    for (auto operand : GetUsedExternalValue(*op)) {
      if (!operand || !operand.defining_op()) {
        continue;
      }
      auto* defined_op = operand.defining_op();
      --pending_count[defined_op];
      if (defined_op && pending_count[defined_op] == 0 &&
          defined_op->GetParent() == block) {
        queue.push(defined_op);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      block->size(),
      sort_ops.size(),
      common::errors::InvalidArgument("sort_ops.size() must be equal to "
                                      "block.size(), but received %d != %d",
                                      block->size(),
                                      sort_ops.size()));

  return sort_ops;
}

std::vector<pir::Operation*> GetProducerOpsReverseSort(
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> producers;

  std::vector<pir::Operation*> vec_res;
  for (auto operand : GetUsedExternalValue(*op)) {
    if (!operand || !operand.defining_op()) {
      continue;
    }
    auto* source_op = operand.defining_op();
    if (source_op && !producers.count(source_op) &&
        source_op->GetParent() == op->GetParent()) {
      producers.insert(source_op);
      PADDLE_ENFORCE(
          op2id.count(source_op),
          common::errors::PreconditionNotMet("source op MUST in op2id map"));
      vec_res.emplace_back(source_op);
    }
  }

  std::sort(vec_res.begin(),
            vec_res.end(),
            [&op2id](pir::Operation* a, pir::Operation* b) {
              return op2id.at(a) > op2id.at(b);
            });

  return vec_res;
}

std::unordered_set<pir::Operation*> GetProducerOps(pir::Operation* op) {
  std::unordered_set<pir::Operation*> producers;

  for (auto operand : GetUsedExternalValue(*op)) {
    if (!operand || !operand.defining_op()) {
      continue;
    }
    auto* source_op = operand.defining_op();
    if (source_op && source_op->GetParent() == op->GetParent()) {
      producers.insert(source_op);
    }
  }
  return producers;
}

std::vector<pir::Operation*> GetProducerOpsRecursive(
    pir::Operation* root,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> visited;
  std::deque<pir::Operation*> queue;
  std::vector<pir::Operation*> result;
  queue.push_back(root);
  visited.insert(root);
  while (!queue.empty()) {
    pir::Operation* cur = queue.front();
    queue.pop_front();
    result.push_back(cur);
    for (const auto& new_op : GetProducerOps(cur)) {
      if (visited.count(new_op)) continue;
      visited.insert(new_op);
      queue.push_back(new_op);
    }
  }
  return result;
}

std::unordered_set<pir::Operation*> GetConsumerOps(
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> consumers;

  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      auto parent_op = it->owner();
      while (parent_op) {
        if (op2id.count(parent_op)) {
          consumers.insert(parent_op);
          break;
        }
        parent_op = parent_op->GetParentOp();
      }
    }
  }
  return consumers;
}

std::unordered_set<pir::Operation*> GetConsumerOpsSimple(pir::Operation* op) {
  std::unordered_set<pir::Operation*> consumers;
  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      auto parent_op = it->owner();
      consumers.insert(parent_op);
    }
  }
  return consumers;
}

std::vector<pir::Operation*> GetConsumerOpsRecursive(
    pir::Operation* root,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> visited;
  std::deque<pir::Operation*> queue;
  std::vector<pir::Operation*> result;
  queue.push_back(root);
  visited.insert(root);
  while (!queue.empty()) {
    pir::Operation* cur = queue.front();
    queue.pop_front();
    result.push_back(cur);
    for (const auto& new_op : GetConsumerOpsSimple(cur)) {
      if (visited.count(new_op)) continue;
      visited.insert(new_op);
      queue.push_back(new_op);
    }
  }
  return result;
}

static std::string OpsDebugStr(std::vector<pir::Operation*> ops) {
  std::stringstream ss;
  pir::IrPrinter printer(ss);
  for (const auto* op : ops) {
    printer.PrintOperation(const_cast<pir::Operation*>(op));
    ss << "{" << op->id() << "}\n";
  }
  return ss.str();
}

struct SubGraph {
  // construct function
  SubGraph() = default;
  // construct function
  SubGraph(pir::Operation* op, bool subst) : substitute(subst) { Insert(op); }
  SubGraph(const std::unordered_set<pir::Operation*>& op, bool subst) {
    substitute = subst;
    for (auto& item : op) {
      Insert(item);
    }
  }
  void Insert(pir::Operation* op) {
    ops.push_back(op);
    op_set.insert(op);

    auto producers = GetProducerOps(op);
    for (auto producer : producers) {
      input_ops.insert(producer);
    }
    input_ops.erase(op);
  }

  void Print() const {
    VLOG(4) << "SubGraph is: " << this;
    VLOG(4) << "=============" << this;
    VLOG(4) << OpsDebugStr(ops);
  }

  int depth{0};
  int max_depth{0};
  int min_depth{INT_MAX};
  bool substitute{true};
  std::vector<pir::Operation*> ops;
  std::unordered_set<pir::Operation*> op_set;
  std::unordered_set<pir::Operation*> input_ops;

  std::unordered_set<SubGraphPtr> producers;
  std::unordered_set<SubGraphPtr> consumers;
};

using OpClassifier = std::function<bool(const pir::Operation&)>;

SubgraphDetector::SubgraphDetector(pir::Block* block,
                                   const OpClassifier& classifier)
    : block_(block), op_classifier_(classifier) {
  sort_ops_ = InverselyTopologicalSort(block_);
  size_t index = 0;
  for (auto& op : *block) {
    op2id_[&op] = index++;
  }
}

std::vector<GroupOpsVec> SubgraphDetector::operator()() {
  DoOpFusion();
  VLOG(4) << "Subgraph list size: " << subgraph_list_.size();
  BuildSubGraph();
  VLOG(4) << "Subgraph list size: " << subgraph_list_.size();

  std::vector<GroupOpsVec> groups;
  for (auto& subgraph : subgraph_list_) {
    if (!subgraph->substitute) {
      continue;
    }

    // sort group ops by natural increasing index.
    std::vector<pir::Operation*> tmp_ops(subgraph->ops.begin(),
                                         subgraph->ops.end());
    auto& op2id = op2id_;
    std::sort(tmp_ops.begin(),
              tmp_ops.end(),
              [&op2id](pir::Operation* a, pir::Operation* b) {
                return op2id.at(a) < op2id.at(b);
              });

    groups.push_back(tmp_ops);
  }

  return groups;
}

using GraphSet = std::unordered_set<SubGraphPtr>;
static GraphSet Union(const GraphSet& upstream, const GraphSet& downstream) {
  GraphSet unioned_set = upstream;
  unioned_set.insert(downstream.begin(), downstream.end());
  return unioned_set;
}

struct UnionFindSet {
  std::unordered_map<pir::Operation*, pir::Operation*> parent;
  std::unordered_map<pir::Operation*, SubGraphPtr> root2subgraph;
  OpClassifier op_classifier_;
  SubGraphPtr GetSetFromGraph(SubGraphPtr x) { return GetSetFromOp(x->ops[0]); }

  pir::Operation* Find(pir::Operation* x) {
    if (parent.find(x) == parent.end()) {
      parent[x] = x;
      return x;
    }
    if (parent[x] != x) {
      parent[x] = Find(parent[x]);
    }
    return parent[x];
  }

  SubGraphPtr Union(pir::Operation* x, pir::Operation* y) {
    auto root_x = Find(x);
    auto root_y = Find(y);
    if (root_x == root_y) {
      return GetSetFromOp(root_y);
    }
    auto subgraph_x = GetSetFromOp(root_x);
    auto subgraph_y = GetSetFromOp(root_y);
    parent[root_x] = root_y;
    // union root_x and root_y;
    for (auto& op : subgraph_x->ops) {
      subgraph_y->Insert(op);
    }
    return subgraph_y;
  }

  SubGraphPtr GetSetFromOp(pir::Operation* op) {
    const auto& root = Find(op);
    if (!root2subgraph.count(root)) {
      root2subgraph[root] = std::make_shared<SubGraph>(op, op_classifier_(*op));
    }
    return root2subgraph[root];
  }

  std::unordered_set<SubGraphPtr> GetUpstream(const SubGraphPtr& graph) {
    std::unordered_set<SubGraphPtr> results;
    for (auto op : graph->op_set) {
      auto producers = GetProducerOps(op);
      for (auto producer : producers) {
        results.insert(GetSetFromOp(producer));
      }
    }
    if (results.count(graph)) {
      results.erase(graph);
    }
    return results;
  }

  std::unordered_set<SubGraphPtr> GetDownstream(const SubGraphPtr& graph) {
    std::unordered_set<SubGraphPtr> results;
    for (auto op : graph->op_set) {
      auto consumers = GetConsumerOpsSimple(op);
      for (auto consumer : consumers) {
        results.insert(GetSetFromOp(consumer));
      }
    }
    if (results.count(graph)) {
      results.erase(graph);
    }
    return results;
  }
};

static GraphSet Intersect(const GraphSet& upstream,
                          const GraphSet& downstream) {
  GraphSet intersected_set;
  for (auto& item : upstream) {
    if (downstream.count(item)) {
      intersected_set.insert(item);
    }
  }
  return intersected_set;
}

struct LoopDetectionMapping {
  std::unordered_map<SubGraphPtr, std::unordered_set<SubGraphPtr>> upstreams_;
  std::unordered_map<SubGraphPtr, std::unordered_set<SubGraphPtr>> downstreams_;
  std::unordered_set<SubGraphPtr> all_nodes_;
  UnionFindSet* uf_set_;
  LoopDetectionMapping(const std::vector<pir::Operation*> sort_ops,
                       const std::unordered_map<pir::Operation*, size_t>& op2id,
                       UnionFindSet* uf_set) {
    for (auto* op : sort_ops) {
      auto producers = GetProducerOpsRecursive(op, op2id);
      auto consumers = GetConsumerOpsRecursive(op, op2id);
      auto op_set = uf_set->GetSetFromOp(op);
      all_nodes_.insert(op_set);
      for (auto producer : producers) {
        auto producer_set = uf_set->GetSetFromOp(producer);
        upstreams_[op_set].insert(producer_set);
      }
      for (auto consumer : consumers) {
        auto consumer_set = uf_set->GetSetFromOp(consumer);
        downstreams_[op_set].insert(consumer_set);
      }
    }
    uf_set_ = uf_set;
  }

  void MergeNodes(const SubGraphPtr& first,
                  const SubGraphPtr& second,
                  const SubGraphPtr& merged) {
    std::unordered_set<SubGraphPtr> merged_upstreams;
    std::unordered_set<SubGraphPtr> merged_downstreams;
    for (auto& item : GetUpstreamSet(first)) merged_upstreams.insert(item);
    for (auto& item : GetUpstreamSet(second)) merged_upstreams.insert(item);
    for (auto& item : GetDownstreamSet(first)) merged_downstreams.insert(item);
    for (auto& item : GetDownstreamSet(second)) merged_downstreams.insert(item);
    upstreams_[merged] = merged_upstreams;
    downstreams_[merged] = merged_downstreams;
    if (first != merged) {
      upstreams_.erase(first);
      downstreams_.erase(first);
      all_nodes_.erase(first);
    }
    if (second != merged) {
      upstreams_.erase(second);
      downstreams_.erase(second);
      all_nodes_.erase(second);
    }
    all_nodes_.insert(merged);
  }
  bool CanFuse(const SubGraphPtr& up, const SubGraphPtr& down) {
    if (up == down) return false;
    GraphSet after_fuse_upstreams =
        Union(GetUpstreamSet(up), GetUpstreamSet(down));
    GraphSet after_fuse_downstreams =
        Union(GetDownstreamSet(up), GetDownstreamSet(down));
    auto intersection = Intersect(after_fuse_upstreams, after_fuse_downstreams);
    intersection.erase(up);
    intersection.erase(down);
    return intersection.size() == 0;
  }

  GraphSet GetUpstreamSet(const SubGraphPtr& cur) {
    GraphSet res;
    for (auto& raw_node : upstreams_[cur]) {
      auto node = uf_set_->GetSetFromGraph(raw_node);
      if (all_nodes_.count(node) && node != cur) res.insert(node);
    }
    upstreams_[cur] = res;
    return res;
  }

  GraphSet GetDownstreamSet(const SubGraphPtr& cur) {
    GraphSet res;
    for (auto& raw_node : downstreams_[cur]) {
      auto node = uf_set_->GetSetFromGraph(raw_node);
      if (all_nodes_.count(node) && node != cur) res.insert(node);
    }
    downstreams_[cur] = res;
    return res;
  }
};

static GraphSet SetDifference(const GraphSet& upstream,
                              const GraphSet& downstream) {
  GraphSet diff_set;
  for (auto& item : upstream) {
    if (!downstream.count(item)) {
      diff_set.insert(item);
    }
  }
  return diff_set;
}

struct RecursiveOpGetter {
  std::unordered_map<SubGraphPtr, GraphSet> memory{};
  std::function<GraphSet(const SubGraphPtr&)> func;
  explicit RecursiveOpGetter(std::function<GraphSet(const SubGraphPtr&)> f) {
    func = f;
  }
  GraphSet operator()(const SubGraphPtr& graph) {
    if (memory.count(graph)) {
      return memory[graph];
    }
    memory[graph] = GraphSet();  // set empty set avoid loop.
    GraphSet set = func(graph);
    GraphSet copied_set = set;
    for (const auto& g : copied_set) {
      auto recur_set = this->operator()(g);
      set.insert(recur_set.begin(), recur_set.end());
    }
    memory[graph] = set;
    return set;
  }
};

static GraphSet UpstreamSet(UnionFindSet& union_set,  // NOLINT
                            const SubGraphPtr& graph) {
  // memory search:
  std::function<GraphSet(const SubGraphPtr&)> GetProducerGraphs =
      [&union_set](const SubGraphPtr& graph) {
        return union_set.GetUpstream(graph);
      };
  auto GetProducerOpsRecursive = RecursiveOpGetter(GetProducerGraphs);
  return GetProducerOpsRecursive(graph);
}

static GraphSet DownstreamSet(UnionFindSet& union_set,  // NOLINT
                              const SubGraphPtr& graph) {
  // memory search:
  std::function<GraphSet(const SubGraphPtr&)> GetConsumerGraphs =
      [&union_set](const SubGraphPtr& graph) {
        return union_set.GetDownstream(graph);
      };
  auto GetConsumerOpsRecursive = RecursiveOpGetter(GetConsumerGraphs);
  return GetConsumerOpsRecursive(graph);
}

static void VLOG_LINES(const std::string& str) {
  if (!VLOG_IS_ON(4)) return;
#ifdef PADDLE_WITH_CINN
  const auto& lines = cinn::utils::Split(str, "\n");
  for (const auto& line : lines) {
    VLOG(4) << line;
  }
#endif
  return;
}

void MergeSubGraphs(Operation* op,
                    Operation* producer,
                    UnionFindSet& union_find,            // NOT NOLINT
                    LoopDetectionMapping& loop_detector  // NOT NOLINT
) {
  if (union_find.GetSetFromOp(op) == union_find.GetSetFromOp(producer)) {
    return;
  }
  if (!loop_detector.CanFuse(union_find.GetSetFromOp(producer),
                             union_find.GetSetFromOp(op))) {
    return;
  }
  // try fuse producer to sub-graph
  auto op_graph_ptr = union_find.GetSetFromOp(op);
  auto producer_graph_ptr = union_find.GetSetFromOp(producer);
  union_find.Union(op, producer);
  loop_detector.MergeNodes(
      op_graph_ptr, producer_graph_ptr, union_find.GetSetFromOp(op));
}

void SubgraphDetector::DoOpFusion() {
  // do fusion
  VLOG(4) << "DoOpFusion";
  UnionFindSet union_find;
  union_find.op_classifier_ = op_classifier_;
  VLOG(4) << "Do Op Fusion with sorted_ops: " << sort_ops_.size();
  VLOG_LINES(OpsDebugStr(sort_ops_));
  LoopDetectionMapping loop_detector(sort_ops_, op2id_, &union_find);

  for (auto* op : sort_ops_) {
    auto producers = GetProducerOpsReverseSort(op, op2id_);
    for (auto* producer : producers) {
      if (!op_classifier_(*op) || !op_classifier_(*producer)) {
        continue;
      }
      VLOG(4) << "Start Judge: " << op->id() << " vs " << producer->id();

      MergeSubGraphs(producer, op, union_find, loop_detector);
    }
  }
  // TODO(chenxi67): Redo this part after bug issue about llama2 is fixed
  // for (auto* op : sort_ops_) {
  //   auto producers = GetProducerOpsReverseSort(op, op2id_);
  //   for (auto* producer : producers) {
  //     if (op_classifier_(*op) && !op_classifier_(*producer)) {
  //       for (auto* consumer : GetConsumerOpsSimple(producer)) {
  //         if (op_classifier_(*consumer) &&
  //             consumer->GetParent() == op->GetParent()) {
  //           MergeSubGraphs(op, consumer, union_find, loop_detector);
  //         }
  //       }
  //       continue;
  //     }
  //   }
  // }
  for (const auto& op : sort_ops_) {
    subgraph_map_[op] = union_find.GetSetFromOp(op);
  }

  for (auto& subgraph : subgraph_map_) {
    auto* op = subgraph.first;
    auto* subgraph_ptr = subgraph.second.get();
    if (union_find.Find(op) == op) {
      VLOG(4) << "Subgraph: " << subgraph_ptr;
      VLOG(4) << "   substitute: " << subgraph_ptr->substitute;
      for (auto& op : subgraph_ptr->ops) {
        VLOG(4) << "ops: " << op->name() << ", " << op->id();
      }
    }
  }
}

void SubgraphDetector::BuildSubGraph() {
  std::unordered_set<SubGraph*> subgraph_set;
  for (auto* op : sort_ops_) {
    PADDLE_ENFORCE_EQ(
        subgraph_map_.count(op),
        true,
        common::errors::InvalidArgument("subgraph_map_ MUST contain op"));
    auto& subgraph = subgraph_map_[op];
    if (subgraph_set.count(subgraph.get())) {
      continue;
    }

    subgraph_set.insert(subgraph.get());
    subgraph_list_.push_back(subgraph);
  }

  for (auto& subgraph : subgraph_list_) {
    for (auto& input_op : subgraph->input_ops) {
      PADDLE_ENFORCE_EQ(
          subgraph_map_.count(input_op),
          true,
          common::errors::InvalidArgument("subgraph_map_ MUST contain op"));
      auto& producer = subgraph_map_[input_op];
      subgraph->producers.insert(producer);
      producer->consumers.insert(subgraph);
    }
  }

  // init group depth.
  for (auto& subgraph : subgraph_list_) {
    for (auto& consumer : subgraph->consumers) {
      // update depth.
      subgraph->depth = std::max(subgraph->depth, consumer->depth + 1);
    }
    subgraph->max_depth = subgraph->depth;
    subgraph->min_depth = subgraph->depth;
  }

  // reverse to keep fusion group in order.
  std::reverse(subgraph_list_.begin(), subgraph_list_.end());
}
std::vector<pir::Value> AnalysisOutputs(
    const GroupOpsVec& group_ops) {  // NOLINT
  // Get output by ud chain
  std::unordered_set<pir::Value> used_by_outside;
  std::unordered_set<pir::Operation*> op_set(group_ops.begin(),
                                             group_ops.end());

  std::vector<pir::Value> outputs;
  for (auto* op : group_ops) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      auto result = op->result(i);

      for (auto use_iter = result.use_begin(); use_iter != result.use_end();
           ++use_iter) {
        if (!op_set.count(use_iter->owner())) {
          outputs.push_back(result);
          break;
        }
      }
    }
  }

  // NOTE: If all value are not used outside, we mark last op's results
  // as outputs. But keep in mind that is risky.
  if (outputs.size() == 0) {
    for (size_t i = 0; i < group_ops.back()->num_results(); ++i) {
      outputs.push_back(group_ops.back()->result(i));
    }
  }

  return outputs;
}

namespace {

struct IncrementalOrder {
  bool operator()(const pir::Operation* lhs, const pir::Operation* rhs) const {
    PADDLE_ENFORCE_EQ(lhs->GetParent() == rhs->GetParent(),
                      true,
                      common::errors::PreconditionNotMet(
                          "lhs and rhs should have same parent block."));
    auto lhs_iter = lhs->operator Block::ConstIterator();
    auto rhs_iter = rhs->operator Block::ConstIterator();
    auto end_iter = lhs->GetParent()->end();
    while (lhs_iter != end_iter) {
      lhs_iter++;
      if (lhs_iter == rhs_iter) return true;
      if (lhs_iter == end_iter) return false;
    }
    PADDLE_ENFORCE_EQ(
        false,
        true,
        common::errors::InvalidArgument("rhs is not reachable from lhs."));
    return false;
  }
};

std::unordered_set<pir::Operation*> GetUpstreamOpsAfterPosition(
    const pir::Operation* position_op,
    const pir::Block* block,
    pir::Operation* op,
    std::unordered_set<pir::Operation*>* visited_ops) {
  std::unordered_set<pir::Operation*> ops;
  const auto& IsInBlock = [](const pir::Operation* src_op,
                             const pir::Block* block) {
    for (auto& item : *block) {
      if (src_op->id() == item.id()) return true;
    }
    return false;
  };
  std::vector<pir::Value> op_inputs = GetUsedExternalValue(*op);
  for (auto value : op_inputs) {
    if (!value || !value.defining_op()) continue;
    pir::Operation* defining_op = value.defining_op();
    if (visited_ops->count(defining_op)) continue;
    visited_ops->insert(defining_op);
    if (!IsInBlock(defining_op, block)) continue;
    if (IncrementalOrder()(defining_op, position_op)) continue;

    ops.insert(defining_op);
    auto recursive_ops = GetUpstreamOpsAfterPosition(
        position_op, block, defining_op, visited_ops);
    ops.insert(recursive_ops.begin(), recursive_ops.end());
  }
  return ops;
}
}  // namespace

void MoveUpstreamOpBeforeGroup(const GroupOpsVec& group_ops,
                               pir::Block* block,
                               pir::Operation* insert_point_op) {
  const auto moved_ops = [&]() {
    std::set<pir::Operation*, IncrementalOrder> ops_set;
    std::unordered_set<pir::Operation*> visited_ops;
    for (auto& op : group_ops) {
      auto upstream_ops =
          GetUpstreamOpsAfterPosition(insert_point_op, block, op, &visited_ops);
      ops_set.insert(upstream_ops.begin(), upstream_ops.end());
    }
    return ops_set;
  }();

  for (auto& op : moved_ops) {
    if (op == insert_point_op) continue;
    VLOG(4) << "Move " << op->id() << " " << op->name() << " before "
            << insert_point_op->id() << " " << insert_point_op->name();
    op->MoveTo(block, insert_point_op->operator Block::Iterator());
  }
}

pir::Operation* FindInsertPoint(const GroupOpsVec& group_ops,
                                const std::vector<pir::Value>& outputs) {
  // Regard last op as insert position if there are no downstream ops between in
  // group_ops.
  pir::Operation* insert_point_op = group_ops.back();
  auto begin = group_ops.front()->operator Block::ConstIterator();
  auto end = ++(group_ops.back()->operator Block::ConstIterator());
  const std::unordered_set<pir::Value> outputs_set(outputs.begin(),
                                                   outputs.end());
  const std::unordered_set<const pir::Operation*> group_ops_set(
      group_ops.begin(), group_ops.end());

  const auto& IsDownstreamOp = [&](const pir::Operation* op) -> bool {
    if (group_ops_set.find(op) != group_ops_set.end()) return false;
    for (auto& value : GetUsedExternalValue(*op)) {
      if (outputs_set.find(value) != outputs_set.end()) {
        return true;
      }
    }
    return false;
  };
  // Find first downstream op as final insert position.
  for (; begin != end; ++begin) {
    if (IsDownstreamOp(begin)) {
      insert_point_op = begin;
      break;
    }
  }
  return insert_point_op;
}

void ReplaceWithGroupOp(pir::Block* block,
                        const GroupOpsVec& group_ops) {  // NOLINT
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
#ifdef PADDLE_WITH_CINN
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
#endif
#ifdef PADDLE_WITH_DNNL
  ctx->GetOrRegisterDialect<paddle::dialect::OneDNNOperatorDialect>();
#endif
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  const std::vector<pir::Value> outputs = AnalysisOutputs(group_ops);

  // step 1: Analysis and insert group op before insert_point.
  auto* insert_point = FindInsertPoint(group_ops, outputs);
  MoveUpstreamOpBeforeGroup(group_ops, block, insert_point);
  builder.set_insertion_point(insert_point);
  VLOG(6) << "Insert GroupOp after " << insert_point->name();

// step 2: Replace the old op with GroupOp.
#ifdef PADDLE_WITH_CINN

  auto new_group_op = [&]() -> cinn::dialect::GroupOp {
    std::vector<pir::Type> output_types;
    for (auto& value : outputs) output_types.emplace_back(value.type());

    auto group_op = builder.Build<cinn::dialect::GroupOp>(output_types);
    for (auto op : group_ops) {
      op->MoveTo(group_op.block(), group_op.block()->end());
    }
    return group_op;
  }();
#else
  auto new_group_op = [&]() -> pir::GroupOp {
    std::vector<pir::Type> output_types;
    for (auto& value : outputs) output_types.emplace_back(value.type());

    auto group_op = builder.Build<pir::GroupOp>(output_types);
    for (auto op : group_ops) {
      op->MoveTo(group_op.block(), group_op.block()->end());
    }
    return group_op;
  }();
#endif

  // step 3: Replace outputs of inner ops
  const std::vector<pir::Value> group_outs = new_group_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].ReplaceUsesWithIf(group_outs[i],
                                 [&inner_ops](pir::OpOperand op) {
                                   return !inner_ops.count(op.owner());
                                 });
  }

  // step 4: Insert YieldOp for outputs
  builder.SetInsertionPointToBlockEnd(new_group_op.block());
  builder.Build<::pir::YieldOp>(outputs);
}

}  // namespace pir
