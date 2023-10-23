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

#include "paddle/fluid/pir/transforms/build_cinn_pass.h"

#include <algorithm>
#include <queue>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builder.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

#include "paddle/cinn/frontend/op_mapper_registry.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/utils/flags.h"

PD_DECLARE_string(allow_cinn_ops);
PD_DECLARE_string(deny_cinn_ops);

namespace {
using GroupOpsVec = std::vector<pir::Operation*>;
// The delim(`;`) that is used to split the FLAGS_allow_cinn_ops
// & FLAGS_deny_cinn_ops.
constexpr char kDelim[] = ";";
using CompatibleInfo = cinn::hlir::framework::pir::CompatibleInfo;

// OpTransInfo contains informations used to detect subgraphs
// supported by the CINN compiler.
class OpTransInfo {
  using DyOpCondT =
      std::unordered_map<std::string, std::function<bool(pir::Operation*)>>;
  using DeParamCondT =
      std::unordered_map<std::string, std::unordered_set<std::string>>;

 public:
  OpTransInfo() {
    // judgment condition for the dynamic slice
    dynamic_op_cond_.emplace("slice", [](pir::Operation* op) -> bool {
      if (!op->attributes().count("infer_flags")) return false;
      auto infer_flags = op->attributes()
                             .at("infer_flags")
                             .dyn_cast<pir::ArrayAttribute>()
                             .AsVector();
      return std::find_if(
                 infer_flags.begin(), infer_flags.end(), [](pir::Attribute& v) {
                   return v.dyn_cast<pir::Int32Attribute>().data() < 0;
                 }) != infer_flags.end();
    });
    // judgment condition for the dynamic reshape
    dynamic_op_cond_.emplace("reshape", [](pir::Operation* op) -> bool {
      bool shape_from_full = op->dyn_cast<paddle::dialect::ReshapeOp>()
                                 .shape()
                                 .dyn_cast<pir::OpResult>()
                                 .owner()
                                 ->isa<paddle::dialect::FullIntArrayOp>();
      return !shape_from_full;
    });
    // judgment condition for the dynamic expand
    dynamic_op_cond_.emplace("expand", [](pir::Operation* op) -> bool {
      bool shape_from_full = op->dyn_cast<paddle::dialect::ExpandOp>()
                                 .shape()
                                 .dyn_cast<pir::OpResult>()
                                 .owner()
                                 ->isa<paddle::dialect::FullIntArrayOp>();
      return !shape_from_full;
    });
  }

  const DyOpCondT& dynamic_op_cond() const { return dynamic_op_cond_; }

  const DeParamCondT& deny_param_cond() const { return deny_param_cond_; }

  const std::unordered_set<std::string>& default_deny_ops() const {
    return default_deny_ops_;
  }

  // TODO(Aurelius84): Deal with the special ops.
  std::unordered_set<pir::Value> GetDenyValues(const GroupOpsVec& group) const {
    return {};
  }

 private:
  DyOpCondT dynamic_op_cond_;

  DeParamCondT deny_param_cond_{{"batch_norm", {"ReserveSpace"}},
                                {"batch_norm_grad", {"ReserveSpace"}}};

  std::unordered_set<std::string> default_deny_ops_{
      "feed", "fetch", "conv2d", "conv2d_grad"};
};

std::unordered_set<std::string> StringSplit(const std::string& str,
                                            const std::string& delim) {
  std::regex reg(delim);
  std::unordered_set<std::string> elems{
      std::sregex_token_iterator(str.begin(), str.end(), reg, -1),
      std::sregex_token_iterator()};
  elems.erase("");
  return elems;
}

std::string GetDebugInfo(const std::unordered_set<std::string>& names) {
  std::string debug_info = "[";
  for (auto& name : names) {
    debug_info.append(name);
    debug_info.append(", ");
  }
  debug_info.append("]");
  return debug_info;
}

bool IsSupportCinn(pir::Operation* op) {
  auto allow_ops = StringSplit(FLAGS_allow_cinn_ops, kDelim);
  auto deny_ops = StringSplit(FLAGS_deny_cinn_ops, kDelim);
  VLOG(4) << "The allowed Cinn Ops: " << GetDebugInfo(allow_ops);
  VLOG(4) << "The denied Cinn Ops: " << GetDebugInfo(deny_ops);
  // Strip the dialect, like pd_op.abs -> abs
  const auto& op_name = CompatibleInfo::OpName(*op);
  bool registered =
      ::cinn::frontend::OpMapperRegistry::Global()->Find(op_name) != nullptr;

  OpTransInfo trans_info;
  bool is_support = registered && !trans_info.default_deny_ops().count(op_name);
  // if the op type is registered in CINN and allow_ops is not empty, return
  // true only when it is in allow_ops
  if (!allow_ops.empty()) {
    return is_support && allow_ops.count(op_name);
  }
  // if the op type is registered in CINN and deny_ops is not empty, return
  // true only when it is not in deny_ops
  if (!deny_ops.empty()) {
    return is_support && !deny_ops.count(op_name);
  }

  VLOG(4) << op->name() << " is_support: " << is_support << " " << registered;

  // if the user doesn't set FLAGS_allow_cinn_ops and FLAGS_deny_cinn_ops,
  // return true only when it is registered in CINN
  return is_support;
}

std::vector<pir::Operation*> InverselyTopologicalSort(pir::Block* block) {
  std::vector<pir::Operation*> sort_ops;
  std::unordered_map<pir::Operation*, int> pending_count;
  // step 1: initialize pending_cout for defined op
  for (auto* op : *block) {
    if (pending_count.find(op) == pending_count.end()) {
      pending_count[op] = 0;
    }
    for (auto& operand : op->operands()) {
      auto* defined_op = operand.source().dyn_cast<pir::OpResult>().owner();
      if (pending_count.find(defined_op) != pending_count.end()) {
        ++pending_count[defined_op];
      } else {
        pending_count[defined_op] = 1;
      }
    }
  }

  std::queue<pir::Operation*> queue;
  for (auto* op : *block) {
    VLOG(4) << op->name() << " pending_count: " << pending_count[op];
    if (pending_count[op] == 0) {
      queue.push(op);
    }
  }

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    VLOG(4) << "Pop Op: " << op->name();
    sort_ops.push_back(op);
    for (auto& operand : op->operands()) {
      auto* defined_op = operand.source().dyn_cast<pir::OpResult>().owner();
      --pending_count[defined_op];
      if (pending_count[defined_op] == 0) {
        queue.push(defined_op);
      }
    }
  }

  IR_ENFORCE(
      block->size() == sort_ops.size(),
      "sort_ops.size() must be equal to block.size(), but received %d != %d",
      block->size(),
      sort_ops.size());

  return sort_ops;
}

struct SubGraph;
using SubGraphPtr = std::shared_ptr<SubGraph>;

std::unordered_set<pir::Operation*> GetProducerOps(pir::Operation* op) {
  std::unordered_set<pir::Operation*> producers;

  for (auto& operand : op->operands()) {
    auto* source_op = operand.source().dyn_cast<pir::OpResult>().owner();
    producers.insert(source_op);
  }
  return producers;
}

std::unordered_set<pir::Operation*> GetConsumerOps(pir::Operation* op) {
  std::unordered_set<pir::Operation*> consumers;

  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      consumers.insert(it->owner());
    }
  }
  return consumers;
}

struct SubGraph {
  // construct function
  SubGraph() {}
  // construct function
  SubGraph(pir::Operation* op, bool subst) : substitute(subst) { Insert(op); }
  void Insert(pir::Operation* op) {
    ops.push_back(op);
    op_set.insert(op);

    auto producers = GetProducerOps(op);
    for (auto producer : producers) {
      input_ops.insert(producer);
    }
    input_ops.erase(op);
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

class CinnSubgraphDetector {
 public:
  // Tell whether a node is inside a sub-graph.
  using OpClassifier = std::function<bool(pir::Operation*)>;

  CinnSubgraphDetector(pir::Block* block, const OpClassifier& classifier)
      : block_(block), op_classifier_(classifier) {}

  std::vector<GroupOpsVec> operator()() {
    DoOpFusion();
    BuildSubGraph();
    DoSubGraphFusion();
    std::vector<GroupOpsVec> groups;
    for (auto& subgraph : subgraph_list_) {
      if (!subgraph->substitute) {
        continue;
      }
      groups.push_back(subgraph->ops);
    }

    return groups;
  }

 protected:
  // Do Op Fusion
  void DoOpFusion() {
    sort_ops_ = InverselyTopologicalSort(block_);
    // do fusion
    for (auto* op : sort_ops_) {
      auto subgraph = subgraph_map_.count(op)
                          ? subgraph_map_[op]
                          : std::make_shared<SubGraph>(op, op_classifier_(op));
      if (!subgraph_map_.count(op)) {
        subgraph_map_[op] = subgraph;
      }
      auto producers = GetProducerOps(op);

      for (auto* producer : producers) {
        if (op_classifier_(producer) != subgraph->substitute) {
          continue;
        }

        bool can_fused = true;
        auto consumers = GetConsumerOps(producer);
        for (auto consumer : consumers) {
          if (!subgraph->op_set.count(consumer)) {
            can_fused = false;
            break;
          }
        }
        if (!can_fused) {
          continue;
        }
        // fuse producer to sub-graph
        subgraph->Insert(producer);
        subgraph_map_[producer] = subgraph;
      }
    }
  }

  void BuildSubGraph() {
    std::unordered_set<SubGraph*> subgraph_set;
    for (auto* op : sort_ops_) {
      CHECK(subgraph_map_.count(op));
      auto& subgraph = subgraph_map_[op];
      if (subgraph_set.count(subgraph.get())) {
        continue;
      }

      subgraph_set.insert(subgraph.get());
      subgraph_list_.push_back(subgraph);
    }

    for (auto& subgraph : subgraph_list_) {
      for (auto& input_op : subgraph->input_ops) {
        CHECK(subgraph_map_.count(input_op));
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

  // SubGraph Fusion
  void DoSubGraphFusion() {
    while (true) {
      bool update = false;
      for (auto& subgraph : subgraph_list_) {
        // sub graph is not substitute
        if (!subgraph->substitute) {
          continue;
        }
        // do fusion
        update |= FuseSubGraph(subgraph);
      }
      if (!update) {
        break;
      }
    }
  }

  bool FuseSubGraph(SubGraphPtr subgraph_ptr) {
    auto producer = subgraph_ptr;
    auto& consumers = producer->consumers;
    std::vector<SubGraphPtr> candidates;
    for (auto& consumer : consumers) {
      if (!consumer->substitute) {
        continue;
      }
      // fast depency check.
      if (IsDependencySimplify(producer, consumer, consumers)) {
        continue;
      }
      // global depency check.
      if (IsDependency(producer, consumer, consumers)) {
        continue;
      }

      candidates.push_back(consumer);
    }

    if (!candidates.size()) {
      return false;
    }

    // fuse candidate to producer
    for (auto& candidate : candidates) {
      candidate->substitute = false;

      // merge nodes
      producer->ops.insert(
          producer->ops.end(), candidate->ops.begin(), candidate->ops.end());
      producer->op_set.insert(candidate->op_set.begin(),
                              candidate->op_set.end());

      // update bound for check depency
      producer->max_depth = std::max(producer->max_depth, candidate->max_depth);
      producer->min_depth = std::min(producer->min_depth, candidate->min_depth);

      // merge producer/consumer
      producer->producers.insert(candidate->producers.begin(),
                                 candidate->producers.end());
      producer->consumers.insert(candidate->consumers.begin(),
                                 candidate->consumers.end());
      // update producers's consumer
      for (auto& tmp : candidate->producers) {
        if (tmp.get() == producer.get()) {
          continue;
        }
        tmp->consumers.insert(producer);
        tmp->consumers.erase(candidate);
      }
      // update consumers's producer
      for (auto& tmp : candidate->consumers) {
        tmp->producers.insert(producer);
        tmp->producers.erase(candidate);
      }

      // remove candicate in producer/consumer
      producer->producers.erase(candidate);
      producer->consumers.erase(candidate);

      // merge input nodes
      producer->input_ops.insert(candidate->input_ops.begin(),
                                 candidate->input_ops.end());
    }

    // remove input nodes that is in node set
    auto input_ops = producer->input_ops;
    for (auto input_op : input_ops) {
      if (producer->op_set.count(input_op)) {
        producer->input_ops.erase(input_op);
      }
    }

    // remove producer from set.
    producer->producers.erase(producer);
    producer->consumers.erase(producer);

    return true;
  }
  // check exist depency.
  bool IsDependency(const SubGraphPtr& producer_g,
                    const SubGraphPtr& consumer,
                    const std::unordered_set<SubGraphPtr>& consumers) {
    std::queue<SubGraphPtr> candidates;
    candidates.push(consumer);

    std::unordered_set<SubGraphPtr> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (auto& producer : candidate->producers) {
        if (producer.get() == producer_g.get()) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }
  bool IsDependencySimplify(const SubGraphPtr& producer_g,
                            const SubGraphPtr& consumer,
                            const std::unordered_set<SubGraphPtr>& consumers) {
    std::queue<SubGraphPtr> candidates;
    candidates.push(consumer);
    // check upper bound.
    int check_upper_depth = producer_g->max_depth;
    std::unordered_set<SubGraphPtr> visited_set;
    while (!candidates.empty()) {
      auto& candidate = candidates.front();
      candidates.pop();
      for (auto& producer : candidate->producers) {
        if (producer.get() == producer_g.get()) {
          continue;
        }
        if (producer->min_depth > check_upper_depth) {
          continue;
        }
        if (consumers.count(producer)) {
          return true;
        }
        if (!visited_set.count(producer)) {
          visited_set.insert(producer);
          candidates.push(producer);
        }
      }
    }
    return false;
  }

 private:
  pir::Block* block_;
  OpClassifier op_classifier_;

  std::vector<pir::Operation*> sort_ops_;
  std::vector<SubGraphPtr> subgraph_list_;
  std::unordered_map<pir::Operation*, SubGraphPtr> subgraph_map_;
};

std::vector<pir::Value> AnalysisOutputs(GroupOpsVec& group_ops) {  // NOLINT
  std::set<pir::Value> inputs;
  std::set<pir::Value> outputs;
  for (auto* op : group_ops) {
    VLOG(4) << "AnalysisOutputs from " << op->name();
    for (auto& operand : op->operands()) {
      inputs.emplace(operand.source());
    }
    for (auto& result : op->results()) {
      outputs.emplace(result);
    }
  }
  std::vector<pir::Value> results;
  std::set_symmetric_difference(outputs.begin(),
                                outputs.end(),
                                inputs.begin(),
                                inputs.end(),
                                std::back_inserter(results));
  VLOG(3) << "Outputs size for GroupOp " << results.size();
  return results;
}

void ReplaceWithGroupOp(pir::Block* block,
                        GroupOpsVec& group_ops) {  // NOLINT
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  // step 1: Ensure the insert point and create GroupOp here.
  auto* laste_input_op = group_ops.back();
  builder.SetInsertionPointAfter(laste_input_op);
  std::vector<pir::Type> output_types;
  std::vector<pir::Value> outputs = AnalysisOutputs(group_ops);
  for (auto& value : outputs) {
    output_types.emplace_back(value.type());
  }
  // step 2: Replace the old op with GroupOp.
  auto new_group_op = builder.Build<cinn::dialect::GroupOp>(output_types);
  pir::Block* group_block = new_group_op.block();
  for (auto* op : group_ops) {
    op->MoveTo(group_block, group_block->begin());
  }
  // step 3: Insert YieldOp for outputs
  builder.SetInsertionPointToEnd(group_block);
  builder.Build<::pir::YieldOp>(outputs);
  // step 4: Replace outputs of inner ops
  std::vector<pir::OpResult> group_outs = new_group_op->results();
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i].ReplaceAllUsesWith(group_outs[i]);
  }
}

class BuildCinnPass : public pir::Pass {
 public:
  BuildCinnPass() : pir::Pass("build_cinn_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "build_cinn_pass should run on module op.");
    auto* block = module_op.block();

    std::vector<GroupOpsVec> groups =
        CinnSubgraphDetector(block, IsSupportCinn)();
    LOG(INFO) << "--- [build_cinn_pass] detected " << groups.size()
              << " cinn supported subgraphs";
    for (auto& group_ops : groups) {
      VLOG(4) << "current group_ops.size(): " << group_ops.size();
      ReplaceWithGroupOp(block, group_ops);
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};
}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateBuildCinnPass() {
  return std::make_unique<BuildCinnPass>();
}

}  // namespace pir

REGISTER_IR_PASS(build_cinn_pass, BuildCinnPass);
