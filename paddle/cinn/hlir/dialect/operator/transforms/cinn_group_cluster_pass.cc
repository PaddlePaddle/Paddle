// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"

#include "paddle/cinn/hlir/dialect/operator/ir/attribute_storage.h"
#include "paddle/cinn/hlir/dialect/operator/ir/cinn_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/group_merge/op_with_group_merge_util.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/sub_graph_detector.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/pattern_rewrite/pattern_match.h"
#include "paddle/pir/pattern_rewrite/pattern_rewrite_driver.h"

namespace cinn {
namespace dialect {
namespace ir {

namespace {

using cinn::hlir::framework::pir::ScheduleInfoNode;

std::unordered_set<pir::Value> GetInnerGeneValue(
    const std::vector<pir::Operation*>& op_list) {
  std::unordered_set<pir::Value> inner_values;

  for (auto op : op_list) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      inner_values.insert(op->result(i));
    }
  }

  return inner_values;
}

std::unordered_set<::pir::Value> GetListOutsideInput(
    const std::vector<::pir::Operation*>& ops) {
  std::unordered_set<pir::Value> outside_ops;
  auto block_inner_output = GetInnerGeneValue(ops);

  for (auto& op : ops) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      if (!block_inner_output.count(op->operand_source(i)) &&
          !outside_ops.count(op->operand_source(i))) {
        outside_ops.insert(op->operand_source(i));
      }
    }
  }
  return outside_ops;
}

bool IsLastReshape(::pir::Operation* input_op) {
  auto out = input_op->result(0);

  if ((out.use_count() == 1) &&
      (out.first_use().owner()->name() == "cf.yield")) {
    return true;
  }

  return false;
}

struct GroupClusterNode {
  std::vector<::pir::Operation*> ops;
  cinn::hlir::framework::OpPatternKind group_kind{
      cinn::hlir::framework::kElementWise};
  std::vector<int64_t> reduce_axis;
  std::vector<int64_t> loop_ranges;

  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      alignment_schedule_info;

  std::unordered_set<::pir::Value> GetOutsideInput() {
    return GetListOutsideInput(ops);
  }

  std::string DebugStr() {
    std::stringstream ss;
    ::pir::IrPrinter printer(ss);

    ss << "type " << group_kind << "\n";
    ss << "loop range\t";

    for (auto d : loop_ranges) {
      ss << ", " << d;
    }
    ss << "\n";
    ss << "reduce axis \t";
    for (auto d : reduce_axis) {
      ss << ", " << d;
    }
    ss << "\n";

    for (auto op : ops) {
      printer.PrintOperation(op);
      if (alignment_schedule_info.count(op)) {
        for (auto& node : alignment_schedule_info.at(op)) {
          ss << node.DebugStr();
        }
      }
      ss << "\n";
    }

    return ss.str();
  }

  void GenerateOutputValue(
      const std::unordered_set<::pir::Value>& outside_need_value) {
    output_value.clear();
    for (auto& op : ops) {
      if (op->name() == "cf.yield") {
        continue;
      }

      std::unordered_set<::pir::Value> inserted_val;
      for (size_t i = 0; i < op->num_results(); ++i) {
        if (outside_need_value.count(op->result(i))) {
          if (!inserted_val.count(op->result(i))) {
            output_value.push_back(op->result(i));

            inserted_val.insert(op->result(i));
          }
        }
      }
    }
  }

  void MergeNode(const GroupClusterNode& node,
                 const ScheduleInfoNode& sch_node) {
    std::unordered_set<::pir::Operation*> inner_ops(ops.begin(), ops.end());

    if (sch_node.type != "") {
      // all the data need add sch node
      for (auto op : ops) {
        alignment_schedule_info[op].push_back(sch_node);
      }
    }
    for (auto op : node.ops) {
      if (!inner_ops.count(op)) {
        ops.push_back(op);
        // copy align info
        if (node.alignment_schedule_info.count(op)) {
          alignment_schedule_info[op] = node.alignment_schedule_info.at(op);
        }
      }
    }

    if (group_kind < node.group_kind) {
      group_kind = node.group_kind;
    }

    if ((node.group_kind == cinn::hlir::framework::kReduction) ||
        (node.group_kind == cinn::hlir::framework::kBroadcast)) {
      reduce_axis = node.reduce_axis;
      loop_ranges = node.loop_ranges;
    }

    if ((ops.size() == 1) && (ops.front()->isa<cinn::dialect::ReshapeOp>())) {
      loop_ranges = node.loop_ranges;
    }
  }

  std::vector<::pir::Value> output_value;
};

::pir::Operation* ReplaceWithGroupOp(::pir::Block* block,
                                     const ::pir::GroupOpsVec& group_ops,
                                     const GroupClusterNode& node,
                                     ::pir::Operation* insert_op,
                                     ::pir::IrMapping* ir_mapping) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();
  ::pir::Builder builder = ::pir::Builder(ctx, block);
  // step 1: Ensure the insert point and create GroupOp here.
  auto* last_op = group_ops.back();
  builder.SetInsertionPointAfter(insert_op);
  auto output_value = node.output_value;
  auto alignment_schedule_info = node.alignment_schedule_info;
  std::vector<pir::Type> output_types;
  // std::vector<pir::Value> outputs = ::pir::AnalysisOutputs(group_ops);

  //  ::pir::IrMapping ir_mapping;
  for (auto& value : output_value) {
    output_types.emplace_back(value.type());
  }

  ::pir::CloneOptions clone_options(false, true, false);

  std::vector<::pir::Operation*> vec_new_op_list;
  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      new_align_info;

  // get basic group id
  // TODO(phlrain) : need update group id
  std::string group_id;
  for (auto op : group_ops) {
    auto new_op = op->Clone(*ir_mapping, clone_options);
    vec_new_op_list.push_back(new_op);
    if (group_id != "") {
      group_id += "_";
    }
    group_id += new_op->name();

    if (alignment_schedule_info.count(op)) {
      new_align_info[new_op] = alignment_schedule_info.at(op);
    }
  }

  cinn::dialect::GroupInfo group_info({});
  group_info.group_id = group_id;
  group_info.loop_ranges = node.loop_ranges;
  group_info.reduce_axis = node.reduce_axis;
  group_info.op_pattern_kind = node.group_kind;
  group_info.alignment_schedule_info = new_align_info;

  // step 2: Replace the old op with GroupOp.
  auto new_group_op =
      builder.Build<cinn::dialect::GroupOp>(output_types, group_info);
  pir::Block* group_block = new_group_op.block();

  for (auto op : vec_new_op_list) {
    group_block->insert(group_block->end(), op);
  }

  // step 3: Replace outputs of inner ops
  auto group_outs = new_group_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());

  std::vector<::pir::Value> new_output;
  for (size_t i = 0; i < output_value.size(); ++i) {
    new_output.push_back(ir_mapping->Lookup<::pir::Value>(output_value[i]));
  }
  builder.SetInsertionPointToBlockEnd(group_block);
  builder.Build<::pir::YieldOp>(new_output);

  return new_group_op;
}

bool CanFuse(const GroupClusterNode& first,
             const GroupClusterNode& second,
             ScheduleInfoNode* sch_node) {
  if ((second.ops.size() == 1) &&
      (second.ops.front()->isa<cinn::dialect::ReshapeOp>()) &&
      (IsLastReshape(second.ops.front()))) {
    return true;
  }

  if ((first.group_kind == cinn::hlir::framework::kReduction &&
       second.group_kind == cinn::hlir::framework::kElementWise) ||
      (first.group_kind == cinn::hlir::framework::kReduction &&
       second.group_kind == cinn::hlir::framework::kBroadcast)) {
    if (first.loop_ranges == second.loop_ranges) {
      return true;
    }
    std::set<int64_t> reduce_axis;
    for (auto axis : first.reduce_axis) {
      if (axis < 0) {
        axis += first.loop_ranges.size();
      }

      reduce_axis.insert(axis);
    }

    if (*(reduce_axis.begin()) !=
        first.loop_ranges.size() - first.reduce_axis.size()) {
      return false;
    }
    if ((first.loop_ranges.size() != second.loop_ranges.size()) &&
        (first.loop_ranges.size() !=
         second.loop_ranges.size() + first.reduce_axis.size())) {
      return false;
    }
    size_t second_index = 0;
    for (size_t i = 0; i < first.loop_ranges.size(); ++i) {
      if (!reduce_axis.count(i)) {
        if (first.loop_ranges[i] != second.loop_ranges[second_index++]) {
          return false;
        }
      } else {
        if (first.loop_ranges.size() == second.loop_ranges.size()) {
          if ((second.loop_ranges[second_index++] != 1)) {
            return false;
          }
        }
      }
    }

    if (first.loop_ranges != second.loop_ranges) {
      sch_node->type = "broadcast";
      sch_node->axis_info = first.reduce_axis;
      sch_node->factor_info = first.loop_ranges;
    }
    return true;
  }

  return (first.loop_ranges == second.loop_ranges) &&
         (first.reduce_axis == second.reduce_axis);
}

std::vector<int> SortNodeList(std::vector<GroupClusterNode>* node_list_ptr,
                              std::vector<std::vector<int>>* pre_ids_ptr) {
  auto& node_list = *node_list_ptr;
  auto& pre_ids = *pre_ids_ptr;
  std::unordered_set<::pir::Value> all_ouput_values;
  for (auto& node : node_list) {
    auto node_outside_input = node.GetOutsideInput();
    all_ouput_values.insert(node_outside_input.begin(),
                            node_outside_input.end());
  }

  for (auto& node : node_list) {
    node.GenerateOutputValue(all_ouput_values);
  }

  std::vector<std::vector<int>> next_ids;
  next_ids.resize(node_list.size());
  for (int i = 0; i < node_list.size(); ++i) {
    for (int j = 0; j < node_list.size(); ++j) {
      if (i == j) {
        continue;
      }

      auto pre_out_list = node_list[i].output_value;
      auto next_in_set = node_list[j].GetOutsideInput();

      for (auto val : pre_out_list) {
        if (next_in_set.count(val)) {
          next_ids[i].push_back(j);
          break;
        }
      }
    }
  }

  std::vector<int> in_degree(next_ids.size(), 0);

  pre_ids.resize(next_ids.size());
  for (int i = 0; i < next_ids.size(); ++i) {
    for (int j = 0; j < next_ids[i].size(); ++j) {
      in_degree[next_ids[i][j]]++;

      pre_ids[next_ids[i][j]].push_back(i);
    }
  }

  std::vector<int> out_id_list;
  std::stack<int> id_stack;
  for (size_t i = 0; i < in_degree.size(); ++i) {
    if (in_degree[i] == 0) {
      id_stack.push(i);
    }
  }

  while (!id_stack.empty()) {
    auto top_id = id_stack.top();
    out_id_list.push_back(top_id);
    id_stack.pop();

    for (auto next_id : next_ids[top_id]) {
      in_degree[next_id]--;

      if (in_degree[next_id] == 0) {
        id_stack.push(next_id);
      }
    }
  }

  if (out_id_list.size() != node_list.size()) {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "out id list size not equal with node list"));
  }

  std::map<int, int> sort_index;
  for (int i = 0; i < out_id_list.size(); ++i) {
    sort_index[out_id_list[i]] = i;
  }

  for (size_t i = 0; i < pre_ids.size(); ++i) {
    std::sort(
        pre_ids[i].begin(), pre_ids[i].end(), [&sort_index](int a, int b) {
          return sort_index.at(a) > sort_index.at(b);
        });
  }

  return out_id_list;
}

void GetClusterNodeInfo(::pir::Operation* op,
                        GroupClusterNode* cluster_node,
                        ScheduleInfoNode* sch_node) {
  cluster_node->group_kind =
      cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op);
  if (cluster_node->group_kind == cinn::hlir::framework::kReduction) {
    // set reduce axis and loop range
    cluster_node->reduce_axis = cinn::dialect::ir::GetVectorAttr(op, "dim");
    cluster_node->loop_ranges =
        phi::vectorize(op->operand_source(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims());
  } else if (cluster_node->group_kind == cinn::hlir::framework::kElementWise) {
    cluster_node->loop_ranges =
        phi::vectorize(op->result(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims());

  } else if (cluster_node->group_kind == cinn::hlir::framework::kBroadcast) {
    cluster_node->loop_ranges =
        phi::vectorize(op->result(0)
                           .type()
                           .dyn_cast<paddle::dialect::DenseTensorType>()
                           .dims());

    sch_node->type = "broadcast";
    sch_node->axis_info =
        cinn::dialect::ir::GetVectorAttr(op, "broadcast_axes");
    sch_node->factor_info = cinn::dialect::ir::GetVectorAttr(op, "out_shape");
  } else {
    std::cerr << "cluter node " << op->name() << std::endl;
    throw std::runtime_error("not support op kind yet");
  }
}

std::vector<GroupClusterNode> GroupSplit(::pir::Operation* input_op) {
  cinn::dialect::GroupOp group_op =
      input_op->dyn_cast<cinn::dialect::GroupOp>();
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  auto target = cinn::common::DefaultNVGPUTarget();

  auto inner_values = GetInnerGeneValue(group_op.GetOperators());

  std::unordered_map<::pir::Operation*, GroupClusterNode> op_path;

  auto op_list = group_op.GetOperators();

  std::vector<GroupClusterNode> first_stage_output;

  std::unordered_set<::pir::Operation*> yield_output_ops;
  auto yield_op = op_list.back();
  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    if (yield_op->operand_source(i).defining_op()->result(0).use_count() == 1) {
      yield_output_ops.insert(yield_op->operand_source(i).defining_op());
    }
  }

  for (auto* op : op_list) {
    if (op->isa<::pir::YieldOp>()) {
      continue;
    }

    auto& cluster_node = op_path[op];
    auto& op_list = cluster_node.ops;

    // process cluster node
    ScheduleInfoNode sch_node;
    GetClusterNodeInfo(op, &cluster_node, &sch_node);

    for (size_t i = 0; i < op->num_operands(); ++i) {
      if (!inner_values.count(op->operand_source(i))) {
        continue;
      }

      auto pre_op = op->operand_source(i).defining_op();

      if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*pre_op) ==
          cinn::hlir::framework::kReduction) {
        continue;
      }

      if (!op_path.count(pre_op)) {
        continue;
      }

      if ((cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) !=
           cinn::hlir::framework::kBroadcast) &&
          (cluster_node.loop_ranges != op_path[pre_op].loop_ranges)) {
        // can not merge for now
        first_stage_output.push_back(op_path[pre_op]);
        continue;
      }
      // get pre op all op list

      std::unordered_set<::pir::Operation*> op_all_set(op_list.begin(),
                                                       op_list.end());
      for (auto inner_op : op_path[pre_op].ops) {
        if (!op_all_set.count(inner_op)) {
          op_list.push_back(inner_op);

          op_all_set.insert(inner_op);
        }
      }

      if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) ==
          cinn::hlir::framework::kBroadcast) {
        for (auto op : op_list) {
          cluster_node.alignment_schedule_info[op].push_back(sch_node);
        }
      }

      cluster_node.alignment_schedule_info.insert(
          op_path[pre_op].alignment_schedule_info.begin(),
          op_path[pre_op].alignment_schedule_info.end());

      if (op_path[pre_op].group_kind > cluster_node.group_kind) {
        cluster_node.group_kind = op_path[pre_op].group_kind;
      }
    }

    op_list.push_back(op);

    if (yield_output_ops.count(op) ||
        cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) ==
            cinn::hlir::framework::kReduction) {
      first_stage_output.push_back(op_path[op]);
    }
  }

  if (first_stage_output.size() <= 1) {
    return first_stage_output;
  }
  // stage 2 merge
  // for now we merge node in same pass
  // only for vertial fuse
  std::vector<GroupClusterNode> second_stage_output = first_stage_output;
  while (true) {
    bool fused = false;
    std::vector<GroupClusterNode> temp_out;

    std::set<int> fused_index;

    std::vector<std::vector<int>> pre_ids_info;
    auto sort_list = SortNodeList(&second_stage_output, &pre_ids_info);

    std::reverse(sort_list.begin(), sort_list.end());
    for (auto node_index : sort_list) {
      if (fused_index.count(node_index)) {
        continue;
      }
      auto& node = second_stage_output[node_index];
      auto& pre_ids = pre_ids_info[node_index];

      GroupClusterNode new_node = node;

      for (auto pre_id : pre_ids) {
        // get pre id

        if (fused_index.count(pre_id)) {
          continue;
        }

        // can new_node merge with pre_id node
        auto& pre_node = second_stage_output[pre_id];

        ScheduleInfoNode sch_node;
        auto can_fuse = CanFuse(pre_node, new_node, &sch_node);
        if (can_fuse) {
          // merge pre node to new_node
          new_node.MergeNode(pre_node, sch_node);

          fused_index.insert(pre_id);
          fused = true;
        } else {
          temp_out.insert(temp_out.begin(), pre_node);
        }
      }
      temp_out.insert(temp_out.end(), new_node);
    }

    if (temp_out.size() >= second_stage_output.size()) {
      // std::cerr << "temp out size " << temp_out.size() << std::endl;
      // throw std::runtime_error("fuse more node here");

      break;
    }
    second_stage_output.swap(temp_out);
    if (fused == false) {
      break;
    }
  }

  if (second_stage_output.size() == 1) {
    return second_stage_output;
  }

  std::vector<std::vector<int>> pre_ids_info;
  auto out_id_list = SortNodeList(&second_stage_output, &pre_ids_info);

  std::vector<GroupClusterNode> sorted_out;
  for (auto id : out_id_list) {
    sorted_out.push_back(second_stage_output[id]);
  }

  return sorted_out;
}

}  // namespace

class CinnGroupClusterPass : public pir::Pass {
 public:
  CinnGroupClusterPass()
      : pir::Pass("cinn_group_cluster_pass", /*opt_level=*/1) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "cinn_group_cluster_pass should run on module op.");
    auto& block = module_op.block();

    for (auto it = block.begin(); it != block.end();) {
      auto base_it = it;

      ++it;
      if (base_it->isa<cinn::dialect::GroupOp>()) {
        ::pir::IrMapping ir_mapping;
        cinn::dialect::GroupOp group_op =
            base_it->dyn_cast<cinn::dialect::GroupOp>();

        auto group_outside_input = GetListOutsideInput(group_op.GetOperators());
        for (auto val : group_outside_input) {
          ir_mapping.Add(val, val);
        }

        auto split_res = GroupSplit(base_it);
        // need sort split res

        std::unordered_set<::pir::Value> all_ouput_values;
        for (auto& node : split_res) {
          auto node_outside_input = node.GetOutsideInput();
          all_ouput_values.insert(node_outside_input.begin(),
                                  node_outside_input.end());
        }

        size_t index = 0;
        std::unordered_map<pir::Operation*, size_t> op2id;

        for (auto op1 : group_op.GetOperators()) {
          op2id[op1] = index++;
        }

        auto yield_op = group_op.GetOperators().back();
        for (size_t i = 0; i < yield_op->num_operands(); ++i) {
          all_ouput_values.insert(yield_op->operand_source(i));
        }

        ::pir::Operation* insert_point = base_it;
        for (auto& node : split_res) {
          node.GenerateOutputValue(all_ouput_values);
          std::vector<pir::Operation*> tmp_ops(node.ops.begin(),
                                               node.ops.end());

          std::sort(tmp_ops.begin(),
                    tmp_ops.end(),
                    [&op2id](pir::Operation* a, pir::Operation* b) {
                      return op2id.at(a) < op2id.at(b);
                    });

          std::unique(tmp_ops.begin(), tmp_ops.end());

          auto node_outside_input = node.GetOutsideInput();

          insert_point = ReplaceWithGroupOp(
              &block, tmp_ops, node, insert_point, &ir_mapping);

          for (size_t i = 0; i < node.output_value.size(); ++i) {
            ir_mapping.Add(node.output_value[i], insert_point->result(i));
          }

          std::unordered_set<::pir::Value> local_outs(node.output_value.begin(),
                                                      node.output_value.end());

          int local_index = 0;

          std::unordered_map<::pir::Value, size_t> value_order;
          for (size_t i = 0; i < yield_op->num_operands(); ++i) {
            value_order[yield_op->operand_source(i)] = i;
          }

          for (size_t i = 0; i < node.output_value.size(); ++i) {
            if (value_order.count(node.output_value[i])) {
              // replace
              base_it->result(value_order.at(node.output_value[i]))
                  .ReplaceAllUsesWith(insert_point->result(i));
            }
          }
        }

        base_it->Erase();
      }
    }
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<pir::ModuleOp>() && op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateCinnGroupClusterPass() {
  return std::make_unique<CinnGroupClusterPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
