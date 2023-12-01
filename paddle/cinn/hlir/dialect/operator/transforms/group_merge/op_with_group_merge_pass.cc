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

#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_pass.h"

#include <limits.h>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/value.h"

namespace cinn {
namespace dialect {
namespace ir {

std::vector<pir::Operation*> GetProducerOpsReverseSort(
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> producers;

  std::vector<pir::Operation*> vec_res;
  for (auto& operand : op->operands()) {
    if (!operand || !(operand.source())) {
      continue;
    }
    auto* source_op = operand.source().dyn_cast<pir::OpResult>().owner();

    if (!op2id.count(source_op)) {
      continue;
    }
    if (!producers.count(source_op)) {
      producers.insert(source_op);
      PADDLE_ENFORCE(
          op2id.count(source_op),
          phi::errors::PreconditionNotMet("source op MUST in op2id map"));
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

  for (auto& operand : op->operands()) {
    if (!operand || !(operand.source())) {
      continue;
    }
    auto* source_op = operand.source().dyn_cast<pir::OpResult>().owner();
    producers.insert(source_op);
  }
  return producers;
}

std::unordered_set<pir::Operation*> GetConsumerOps(
    pir::Operation* op,
    const std::unordered_map<pir::Operation*, size_t>& op2id) {
  std::unordered_set<pir::Operation*> consumers;

  for (auto& result : op->results()) {
    for (auto it = result.use_begin(); it != result.use_end(); ++it) {
      if (!op2id.count(it->owner())) {
        continue;
      }
      consumers.insert(it->owner());
    }
  }
  return consumers;
}

std::vector<pir::Operation*> TopologicalSort(
    const std::vector<::pir::Operation*>& op_list) {
  std::vector<pir::Operation*> sort_ops;
  std::unordered_map<pir::Operation*, int> pending_count;
  // step 1: initialize pending_cout for defined op
  std::unordered_set<pir::Operation*> inner_set(op_list.begin(), op_list.end());

  for (auto* op : op_list) {
    int count = 0;
    for (auto& operand : op->operands()) {
      if (!operand || !(operand.source())) {
        continue;
      }

      if (inner_set.count(operand.source().dyn_cast<pir::OpResult>().owner())) {
        count++;
      }
    }

    pending_count[op] = count;
  }

  std::stack<pir::Operation*> queue;
  for (auto* op : op_list) {
    VLOG(4) << op->name() << " pending_count: " << pending_count[op];
    if (pending_count[op] == 0) {
      queue.push(op);
    }
  }

  while (!queue.empty()) {
    auto* op = queue.top();
    queue.pop();
    VLOG(4) << "Pop Op: " << op->name();
    sort_ops.push_back(op);

    for (auto& result : op->results()) {
      if (!result) {
        continue;
      }

      for (auto it = result.use_begin(); it != result.use_end(); ++it) {
        auto* next_op = (*it).owner();
        if (!pending_count.count(next_op)) {
          continue;
        }
        --pending_count[next_op];
        if (pending_count[next_op] == 0) {
          queue.push(next_op);
        }
      }
    }
  }

  return sort_ops;
}

phi::DDim GetFirstInputShape(const ::pir::Operation* op) {
  if (op->num_operands() == 0) {
    return phi::DDim({});
  }
  auto in = op->operand_source(0);

  return in.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

phi::DDim GetValueShape(const ::pir::Value& value) {
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

bool WithoutLastDimInReduce(const std::vector<int64_t>& inshape,
                            const std::vector<int64_t>& axes) {
  // if last axis is in reduce.
  if (std::find(axes.begin(), axes.end(), inshape.size() - 1) != axes.end() ||
      std::find(axes.begin(), axes.end(), -1) != axes.end()) {
    return false;
  }

  int64_t sum_last_axes = 1;
  for (size_t idx = axes.back() + 1; idx < inshape.size(); ++idx) {
    sum_last_axes *= inshape[idx];
  }

  if (sum_last_axes > 1) {
    return true;
  } else {
    return false;
  }
}

int GetSharedSize(::pir::Operation* op) {
  auto inshape = ::common::vectorize<int64_t>(GetValueShape(op->result(0)));

  auto axes = GetVectorAttr(op, "dim");

  if (WithoutLastDimInReduce(inshape, axes)) {
    int lane = 1;
    for (size_t idx = axes.back() + 1; idx < inshape.size(); ++idx) {
      lane = inshape[idx];
    }
    // int max_num_threads =
    // cinn::common::DefaultNVGPUTarget().max_num_threads(); todo(phlrain): get
    // gpu max threads
    int max_num_threads = 2048;
    if (lane > max_num_threads / 2) {
      return 0;
    }
    int index = axes.size() - 1;
    for (; index >= 0; --index) {
      if (static_cast<size_t>(index + 1) < axes.size() &&
          axes[index] != axes[index + 1] - 1) {
        break;
      }
      lane *= inshape[axes[index]];
      if (lane > max_num_threads / 2) {
        break;
      }
    }
    // if lane > (max_num_threads / 2),the loop break from lane >
    // max_num_threads / 2.
    int axis = lane > (max_num_threads / 2) ? axes[index] : axes[index + 1];
    if (lane <= max_num_threads) {
      return lane * sizeof(float);
    } else {
      int prefix = inshape[axis];
      int tail = lane / prefix;
      for (int idx = max_num_threads / tail;
           idx > ((max_num_threads / 2) / tail);
           --idx) {
        if (prefix % idx == 0) {
          return idx * tail * sizeof(float);
        }
      }
      int num = max_num_threads / tail;
      return num * tail * sizeof(float);
    }
  }
  return 0;
}

using ConditionFunction =
    std::function<bool(::pir::Operation*, const GroupPtr&)>;

// Op Fusion Pass which performs Ops fusion, Ops are fused
// "vertically", meaning producing Ops are fused into their consumers
// with the intent that the loops which compute their values will be fused in
// code generation.
class OpFusionPassHelper {
 public:
  explicit OpFusionPassHelper(
      const std::vector<pir::Operation*>& op_list,
      const std::vector<pir::Operation*>& output_op_list = {},
      const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis =
          nullptr)
      : shape_analysis_(shape_analysis) {
    // init fusion relation
    InitFusionRelation();
    // filter op data, create group for each op
    // auto ops_inorder = std::get<0>(graph->topological_order());

    for (auto it = op_list.begin(); it != op_list.end(); ++it) {
      local_ops_.insert(*it);
    }

    for (size_t i = 0; i < output_op_list.size(); ++i) {
      output_ops_set_.insert(output_op_list[i]);
    }

    auto top_sort_list = TopologicalSort(op_list);
    int index = 0;
    std::stringstream ss;
    ::pir::IrPrinter printer(ss);
    for (auto it = top_sort_list.begin(); it != top_sort_list.end(); ++it) {
      auto op = *it;
      if (op) {
        ops_.push_back(op);
        printer.PrintOperation(op);
        ss << "\n";
        auto group = std::make_shared<Group>();
        // init group
        group->ops.push_back(op);
        group->ops_set.insert(op);
        group->output_ops.insert(op);
        // input op

        for (size_t i = 0; i < op->num_operands(); ++i) {
          auto input = op->operand_source(i).dyn_cast<pir::OpResult>().owner();
          if (input && (local_ops_.count(input))) {
            group->input_ops[input] = 1;
          }
        }

        // group type
        group->op_pattern_kind =
            hlir::framework::pir::CompatibleInfo::OpKind(*op);
        // use current op as master op for schedule
        group->master_ops.insert(op);

        // get opration unique id
        group->group_id = "id_" + std::to_string(index++);
        fusion_groups_[op] = group;
      }
    }

    // reverse op for output to input
    std::reverse(ops_.begin(), ops_.end());
    for (size_t i = 0; i < top_sort_list.size(); ++i) {
      op2id_[top_sort_list[i]] = i;
    }
  }

  // return a vector of groups in topological order.
  GroupList operator()(bool do_fusion = true) {
    // do op fusion.
    if (do_fusion) {
      DoOpFusion();
    }

    // find all fusion group.
    GroupList fusion_groups;
    std::unordered_set<Group*> groups_set;
    for (auto op : ops_) {
      auto& group = fusion_groups_[op];
      if (!groups_set.count(group.get())) {
        groups_set.insert(group.get());
        fusion_groups.push_back(group);
        // reverse ops order to producer->consumer.
        std::reverse(group->ops.begin(), group->ops.end());
      }
    }

    // producer consumer
    for (auto& consumer : fusion_groups) {
      for (auto& input_op : consumer->input_ops) {
        if (!local_ops_.count(input_op.first)) {
          continue;
        }
        auto& producer = fusion_groups_[input_op.first];
        consumer->mut_producer_groups()->insert(producer);
        producer->mut_consumer_groups()->insert(consumer);
      }
    }

    // init group depth.
    for (auto& group : fusion_groups) {
      for (const auto& consumer : group->consumer_groups()) {
        // update depth.
        group->depth = std::max(group->depth, consumer->depth + 1);
      }
    }

    // reverse to keep fusion group in order.
    std::reverse(fusion_groups.begin(), fusion_groups.end());

    return fusion_groups;
  }

 private:
  void DoOpFusion() {
    for (auto consumer : ops_) {
      auto consumer_kind =
          hlir::framework::pir::CompatibleInfo::OpKind(*consumer);
      // kNonFusible op can't fuse any other op.
      if (consumer_kind == OpPatternKind::kNonFusible) {
        continue;
      }

      // fusion op for consumer
      auto consumer_fusion = fusion_groups_[consumer];  //
      // check all linkin op
      // for (size_t i = 0; i < consumer->num_operands(); ++i) {
      auto producer_list = GetProducerOpsReverseSort(consumer, op2id_);
      for (size_t i = 0; i < producer_list.size(); ++i) {
        // auto producer_data = consumer->operand_source(i);

        auto producer = producer_list[i];
        if (!local_ops_.count(producer)) {
          continue;
        }

        // if producer is fused.
        if (consumer_fusion->ops_set.count(producer)) {
          // VLOG(3) << "Op " << producer->id() << " is fused.";
          continue;
        }
        // if producer data is placeholder
        if (!producer) {
          continue;
        }
        // kNonFusible op can't fuse any other op.
        auto producer_kind =
            hlir::framework::pir::CompatibleInfo::OpKind(*producer);
        if (producer_kind == OpPatternKind::kNonFusible) {
          continue;
        }
        // VLOG(3) << "Producer Op: " << producer->id()
        //         << ", Op Pattern: " << producer_kind
        //         << " -> Consumer Op: " << consumer->id()
        //         << ", Op Pattern: " << consumer_kind;
        bool can_fuse = true;
        // checkout producer op outputs are all in fusion op

        // find all the op use by
        size_t producer_data_used_num = 0;

        auto consumer_list = GetConsumerOps(producer, op2id_);
        for (auto consumer_op : consumer_list) {
          producer_data_used_num++;
          // if fusion group can't find op, can't merge
          if (consumer_fusion->ops_set.find(consumer_op) ==
              consumer_fusion->ops_set.end()) {
            can_fuse = false;
            break;
          }
        }

        if (!can_fuse || !CanFuse(producer, consumer)) {
          continue;
        }

        // VLOG(3) << "Fuse Op " << producer->id() << " into Op "
        //         << consumer->id();

        // fuse producer to fusion group
        // TODO(phrain) : support id
        // consumer_fusion->group_id =
        //     producer->id() + "_" + consumer_fusion->group_id;

        consumer_fusion->group_id = consumer_fusion->group_id;
        consumer_fusion->ops.push_back(producer);
        consumer_fusion->ops_set.insert(producer);
        consumer_fusion->input_ops.erase(producer);
        consumer_fusion->op_pattern_kind =
            static_cast<int>(consumer_fusion->op_pattern_kind) >
                    static_cast<int>(producer_kind)
                ? consumer_fusion->op_pattern_kind
                : producer_kind;

        if (producer_kind == OpPatternKind::kReduction) {
          consumer_fusion->master_ops.insert(producer);
        }

        if (output_ops_set_.count(producer)) {
          // VLOG(3) << "Insert Global Output Node : " << producer->id();
          consumer_fusion->output_ops.insert(producer);
        } else if (producer_data_used_num > 1 && producer->num_operands() > 0 &&
                   is_same_size(producer, consumer_fusion)) {
          // producer is not a const value op.
          consumer_fusion->internal_ops.insert(producer);
        }

        // fuse input op

        auto producer_fusion = fusion_groups_[producer];
        for (auto input_op : producer_fusion->input_ops) {
          if (consumer_fusion->input_ops.count(input_op.first)) {
            consumer_fusion->input_ops[input_op.first] += input_op.second;
          } else {
            consumer_fusion->input_ops.insert(input_op);
          }
        }
        // update op group
        fusion_groups_[producer] = consumer_fusion;
      }
    }
  }

  void InitFusionRelation() {
    // fusion relation.
    // 1.kElementwise as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {OpPatternKind::kElementWise,
                          OpPatternKind::kBroadcast,
                          OpPatternKind::kReduction,
                          OpPatternKind::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Elementwise + *Elementwise*). As
          // has same output shape, can always fuse.
          {OpPatternKind::kElementWise, always_fuse},
          // must be horizontal, as Elementwise + Broadcast is left to fusion
          // merge pass.
          {OpPatternKind::kBroadcast,
           [](::pir::Operation* producer, const GroupPtr& consumer) -> bool {
             // NOTE, producer and consumer NEVER be same size
             if (is_same_size(producer, consumer)) {
               return true;
             }

             // NOTE, original code is below, if produer is not output op,
             // result always be true
             // !helper->output_ops_set_.count(producer);
             return true;
           }},
          // horizontal or vertical relation, check with same output shape with
          // horizontal relation or with last
          // successive dimension less than 1024 for gpu.
          {OpPatternKind::kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or can compute inline.
          {OpPatternKind::kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {OpPatternKind::kOutFusible, is_same_shape}};
      fusion_relation_map_[OpPatternKind::kElementWise] = std::move(relation);
    }
    // 2.kBroadcast as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {OpPatternKind::kElementWise,
                          OpPatternKind::kReduction,
                          OpPatternKind::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Broadcast + *Elementwise*), check
          // with same output shape.
          {OpPatternKind::kElementWise, is_same_size},
          // must be horizontal, as Broadcast + Broadcast is not allowed.
          {OpPatternKind::kBroadcast, is_same_size},
          // horizontal or vertical relation(Broadcast + Reduce).
          {OpPatternKind::kReduction, horizontal_or_vertical_reduce_relation},
          // can be horizontal or can compute inline, check with same output
          // shape or just one consumer.
          {OpPatternKind::kInjective, horizontal_or_can_inline},
          // must be horizontal, check with same output shape.
          {OpPatternKind::kOutFusible, is_same_shape}};
      fusion_relation_map_[OpPatternKind::kBroadcast] = std::move(relation);
    }
    // 3.kReduction as producer
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {OpPatternKind::kElementWise,
                          OpPatternKind::kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation(Reduce + Elementwise*), check
          // without last dimension in reduce.
          {OpPatternKind::kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape and
          // without last dimension in reduce.
          {OpPatternKind::kBroadcast, reduce_fuse_broadcast},
          // must be horizontal relation and with same reduce attr.
          {OpPatternKind::kReduction, reduce_fuse_reduce},
          // no_fuse
          {OpPatternKind::kInjective, no_fuse},
          // can't fuse.
          {OpPatternKind::kOutFusible, no_fuse}};
      fusion_relation_map_[OpPatternKind::kReduction] = std::move(relation);
    }
    // 4.kInjective
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {OpPatternKind::kElementWise,
                          OpPatternKind::kInjective};
      // producer -> fusion
      relation.fusion_op_kind = {
          // can be horizontal or vertical(Injective + Elementwise), check with
          // same output shape.
          {OpPatternKind::kElementWise, is_same_size},
          // must be horizontal relation, check with same output shape.
          {OpPatternKind::kBroadcast, horizontal_with_same_size},
          // left to fusion merge pass.
          {OpPatternKind::kReduction, no_fuse},
          // must be horizontal relation, check with same output shape.
          {OpPatternKind::kInjective, horizontal_or_can_inline},
          // can't fuse.
          {OpPatternKind::kOutFusible, no_fuse},
      };
      fusion_relation_map_[OpPatternKind::kInjective] = std::move(relation);
    }
    // 5.kOutFusible
    {
      FusionRelation relation;
      // producer -> consumer
      relation.op_kind = {OpPatternKind::kElementWise,
                          OpPatternKind::kBroadcast};
      // producer -> fusion
      relation.fusion_op_kind = {
          // horizontal or vertical relation, check has same shape.
          {OpPatternKind::kElementWise, is_same_shape},
          // it must be horizontal relation, check has same shape.
          {OpPatternKind::kBroadcast, is_same_shape},
          // can't fuse.
          {OpPatternKind::kReduction, no_fuse},
          // must be horizontal relation, check has same shape.
          {OpPatternKind::kInjective, is_same_shape},
          // can't fuse.
          {OpPatternKind::kOutFusible, no_fuse},
      };
      fusion_relation_map_[OpPatternKind::kOutFusible] = std::move(relation);
    }
  }

  bool CanFuse(::pir::Operation* producer, const ::pir::Operation* consumer) {
    auto& relation =
        fusion_relation_map_[hlir::framework::pir::CompatibleInfo::OpKind(
            *producer)];
    // first step: check producer can be fused into consumer
    if (relation.op_kind.count(
            hlir::framework::pir::CompatibleInfo::OpKind(*consumer))) {
      auto& consumer_group = fusion_groups_[consumer];
      // second step: check producer can be fused into consumer group
      VLOG(3) << "Call ConditionFunction, Producer Op Pattern : "
              << hlir::framework::pir::CompatibleInfo::OpKind(*producer)
              << " , Consumer Group Pattern : "
              << consumer_group->op_pattern_kind;

      return relation.fusion_op_kind[consumer_group->op_pattern_kind](
          producer, fusion_groups_[consumer]);
    }

    return false;
  }

  std::shared_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis() const {
    return CHECK_NOTNULL(shape_analysis_.lock());
  }

  std::vector<::pir::Operation*> ops_;
  std::unordered_map<const ::pir::Operation*, GroupPtr> fusion_groups_;
  std::unordered_set<const ::pir::Operation*> output_ops_set_;

  std::unordered_map<::pir::Operation*, size_t> op2id_;

  std::vector<std::shared_ptr<Group>> groups_;

  std::unordered_set<const ::pir::Operation*> local_ops_;

  struct FusionRelation {
    // producer -> consumer
    std::unordered_set<OpPatternKind> op_kind = {};
    // producer -> fusion sonsumer
    std::unordered_map<OpPatternKind, ConditionFunction> fusion_op_kind = {};
  };
  std::unordered_map<OpPatternKind, FusionRelation> fusion_relation_map_;
  std::weak_ptr<pir::ShapeConstraintIRAnalysis> shape_analysis_;
};

GroupList OpFusionPassInternal(
    const std::vector<pir::Operation*>& op_list,
    const std::vector<pir::Operation*>& output_op_list,
    const std::shared_ptr<pir::ShapeConstraintIRAnalysis>& shape_analysis) {
  VLOG(3) << "OpFusionPass...!";

  auto op_fusion_helper =
      OpFusionPassHelper(op_list, output_op_list, shape_analysis);
  auto res = op_fusion_helper();

  if (VLOG_IS_ON(6)) {
    std::stringstream ss;
    ::pir::IrPrinter printer(ss);
    for (size_t i = 0; i < res.size(); ++i) {
      auto group = res[i];
      ss << "group\t" << group->group_id << std::endl;
      ss << "kind\t" << group->kind() << std::endl;

      for (auto op : group->ops) {
        printer.PrintOperation(op);
        ss << "\n";
      }
    }
    VLOG(6) << ss.str();
  }
  VLOG(3) << "OpFusionPass Finish...!";

  VLOG(3) << "OpFusionPass Finish...!";

  return res;
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
