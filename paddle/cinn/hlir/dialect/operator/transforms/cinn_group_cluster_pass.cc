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
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pattern_rewrite/pattern_applicator.h"
#include "paddle/pir/include/pattern_rewrite/pattern_match.h"
#include "paddle/pir/include/pattern_rewrite/pattern_rewrite_driver.h"

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

  for (const auto& op : ops) {
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

std::string BuildGroupId(const ::pir::GroupOpsVec& ops_list) {
  std::string group_id;
  for (const auto& op : ops_list) {
    if (group_id != "") {
      group_id += "_";
    }
    group_id += op->name();
  }

  return group_id;
}
struct GroupClusterNode {
  // all the ops in each Node
  std::vector<::pir::Operation*> ops;
  // group kind
  cinn::hlir::framework::OpPatternKind group_kind{
      cinn::hlir::framework::kElementWise};
  // reduce_axis if kind is Reduce else empty
  std::vector<int64_t> reduce_axis;
  // if kind is reduce, loop ranges equal input dim
  // if kind id elementwise or broadcast, loop ranges equal output dim
  std::vector<int64_t> loop_ranges;

  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      alignment_schedule_info;

  std::unordered_set<::pir::Value> GetOutsideInput() const {
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

    for (const auto& op : ops) {
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

  void MergeNode(const GroupClusterNode& node,
                 const ScheduleInfoNode& inner_sch_node) {
    std::unordered_set<::pir::Operation*> inner_ops(ops.begin(), ops.end());

    if (inner_sch_node.type != hlir::framework::pir::ScheduleAlignType::kNone) {
      for (const auto& op : ops) {
        this->alignment_schedule_info[op].push_back(inner_sch_node);
      }
    }
    for (const auto& op : node.ops) {
      if (!inner_ops.count(op)) {
        this->ops.push_back(op);
        // copy align info
        if (node.alignment_schedule_info.count(op)) {
          this->alignment_schedule_info[op] =
              node.alignment_schedule_info.at(op);
        }
      }
    }

    if (this->group_kind < node.group_kind) {
      this->group_kind = node.group_kind;
    }

    if ((node.group_kind == cinn::hlir::framework::kReduction) ||
        (node.group_kind == cinn::hlir::framework::kBroadcast)) {
      this->loop_ranges = node.loop_ranges;
    }
    if (node.group_kind == cinn::hlir::framework::kReduction) {
      this->reduce_axis = node.reduce_axis;
    }

    if ((ops.size() == 1) && (ops.front()->name() == "cinn_op.reshape")) {
      this->loop_ranges = node.loop_ranges;
    }
  }

  void MergePreNode(const GroupClusterNode& node,
                    const ScheduleInfoNode& pre_sch_node) {
    std::unordered_set<::pir::Operation*> inner_ops(ops.begin(), ops.end());

    for (const auto& op : node.ops) {
      if (!inner_ops.count(op)) {
        this->ops.push_back(op);
        // copy align info
        if (node.alignment_schedule_info.count(op)) {
          this->alignment_schedule_info[op] =
              node.alignment_schedule_info.at(op);
        }

        if (pre_sch_node.type !=
            hlir::framework::pir::ScheduleAlignType::kNone) {
          this->alignment_schedule_info[op].push_back(pre_sch_node);
        }
      }
    }

    if (group_kind < node.group_kind) {
      this->group_kind = node.group_kind;
    }
  }
};

std::vector<::pir::Value> GenerateOutputValue(
    const std::vector<::pir::Operation*>& ops,
    const std::unordered_map<::pir::Value, size_t>& outside_need_value) {
  std::vector<::pir::Value> temp_out;
  for (const auto& op : ops) {
    if (op->isa<pir::YieldOp>()) {
      continue;
    }

    std::unordered_set<::pir::Value> inserted_val;
    for (size_t i = 0; i < op->num_results(); ++i) {
      if (outside_need_value.count(op->result(i))) {
        if (!inserted_val.count(op->result(i))) {
          temp_out.push_back(op->result(i));

          inserted_val.insert(op->result(i));
        }
      }
    }
  }
  std::sort(temp_out.begin(),
            temp_out.end(),
            [&outside_need_value](::pir::Value a, ::pir::Value b) {
              return outside_need_value.at(a) < outside_need_value.at(b);
            });

  return temp_out;
}

cinn::dialect::GroupInfo BuildGroupInfo(
    const ::pir::GroupOpsVec& vec_new_op_list,
    const GroupClusterNode& node,
    const std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>&
        new_align_info) {
  cinn::dialect::GroupInfo group_info(vec_new_op_list);
  group_info.group_id = BuildGroupId(vec_new_op_list);
  group_info.loop_ranges = node.loop_ranges;
  group_info.reduce_axis = node.reduce_axis;
  group_info.op_pattern_kind = node.group_kind;
  group_info.alignment_schedule_info = new_align_info;

  return group_info;
}

std::vector<pir::Type> BuildOutType(
    const std::vector<::pir::Value>& output_value) {
  std::vector<pir::Type> output_types;

  for (const auto& value : output_value) {
    output_types.emplace_back(value.type());
  }

  return output_types;
}

::pir::GroupOpsVec CloneOps(
    const ::pir::GroupOpsVec& group_ops,
    const GroupClusterNode& node,
    ::pir::IrMapping* ir_mapping,
    std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>*
        align_info) {
  std::vector<::pir::Operation*> vec_new_op_list;
  ::pir::CloneOptions clone_options(false, true, false);

  auto& alignment_schedule_info = node.alignment_schedule_info;
  for (auto op : group_ops) {
    auto new_op = op->Clone(*ir_mapping, clone_options);
    auto& shape_analysis =
        pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    for (size_t i = 0; i < op->num_results(); ++i) {
      if (shape_analysis.HasShapeOrDataForValue(op->result(i))) {
        shape_analysis.SetShapeOrDataForValue(
            new_op->result(i),
            shape_analysis.GetShapeOrDataForValue(op->result(i)));
      }
    }

    vec_new_op_list.push_back(new_op);

    if (alignment_schedule_info.count(op)) {
      align_info->emplace(new_op, alignment_schedule_info.at(op));
    }
  }

  return vec_new_op_list;
}

::pir::Operation* ReplaceWithGroupOp(
    pir::PatternRewriter* rewriter,
    const ::pir::GroupOpsVec& group_ops,
    const GroupClusterNode& node,
    const std::vector<::pir::Value> output_value,
    ::pir::IrMapping* ir_mapping) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<::pir::ControlFlowDialect>();

  std::unordered_map<::pir::Operation*, std::vector<ScheduleInfoNode>>
      new_align_info;

  auto vec_new_op_list = CloneOps(group_ops, node, ir_mapping, &new_align_info);

  auto group_info = BuildGroupInfo(vec_new_op_list, node, new_align_info);
  // step 2: Replace the old op with GroupOp.

  auto output_types = BuildOutType(output_value);
  auto new_fusion_op =
      rewriter->Build<cinn::dialect::FusionOp>(output_types, group_info);
  pir::Block* fusion_block = new_fusion_op.block();

  for (auto op : vec_new_op_list) {
    fusion_block->insert(fusion_block->end(), op);
  }

  // step 3: Replace outputs of inner ops
  auto group_outs = new_fusion_op->results();
  std::unordered_set<pir::Operation*> inner_ops(group_ops.begin(),
                                                group_ops.end());

  std::vector<::pir::Value> new_output;

  for (size_t i = 0; i < output_value.size(); ++i) {
    new_output.push_back(ir_mapping->Lookup<::pir::Value>(output_value[i]));
  }

  rewriter->SetInsertionPointToBlockEnd(fusion_block);
  rewriter->Build<::pir::YieldOp>(new_output);
  rewriter->SetInsertionPointAfter(new_fusion_op);

  return new_fusion_op;
}

bool CanFuse(const GroupClusterNode& first,
             const GroupClusterNode& second,
             ScheduleInfoNode* sch_node) {
  if (!first.ops.empty() &&
      (first.ops.front()->name() == "cinn_op.generate_shape")) {
    return true;
  }
  if ((second.ops.size() == 1) &&
      (second.ops.front()->name() == "cinn_op.reshape") &&
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
      sch_node->type = hlir::framework::pir::ScheduleAlignType::kBroadcast;
      for (auto& d : first.reduce_axis) {
        if (d < 0) {
          sch_node->axis_info.push_back(d + first.loop_ranges.size());
        } else {
          sch_node->axis_info.push_back(d);
        }
      }
      sch_node->factor_info = first.loop_ranges;
    }
    return true;
  }

  return (first.loop_ranges == second.loop_ranges) &&
         (first.reduce_axis == second.reduce_axis);
}

std::vector<int> SortNodeList(std::vector<GroupClusterNode>* node_list_ptr,
                              std::vector<std::vector<int>>* pre_ids_ptr) {
  // sort node list by topological sort
  // TODO(phlrain): One node may have two pre node, need update here
  auto& node_list = *node_list_ptr;
  auto& pre_ids = *pre_ids_ptr;
  std::unordered_map<::pir::Value, size_t> in_out_values;
  for (const auto& node : node_list) {
    auto node_outside_input = node.GetOutsideInput();
    for (const auto& val : node_outside_input) {
      size_t id = in_out_values.size();
      in_out_values.emplace(val, id);
    }
  }

  std::vector<std::vector<pir::Value>> output_values_list;
  for (const auto& node : node_list) {
    output_values_list.push_back(GenerateOutputValue(node.ops, in_out_values));
  }

  std::vector<std::vector<int>> next_ids;
  next_ids.resize(node_list.size());
  for (int i = 0; i < node_list.size(); ++i) {
    for (int j = 0; j < node_list.size(); ++j) {
      if (i == j) {
        continue;
      }

      const auto& pre_out_list = output_values_list[i];
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
    throw std::runtime_error("id list not match");
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

void GetClusterNodeBasicInfo(::pir::Operation* op,
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
    if (cluster_node->reduce_axis.size() == 0) {
      for (size_t i = 0; i < cluster_node->loop_ranges.size(); ++i) {
        cluster_node->reduce_axis.push_back(i);
      }
    }
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

    sch_node->type = hlir::framework::pir::ScheduleAlignType::kBroadcast;
    sch_node->axis_info =
        cinn::dialect::ir::GetVectorAttr(op, "broadcast_axes");
    sch_node->factor_info = cinn::dialect::ir::GetVectorAttr(op, "out_shape");
  } else if (op->name() == "cinn_op.generate_shape") {
    // do nothing for now
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "only support elementwise, broadcast, reduce type"));
  }
}

std::vector<::pir::Operation*> GetPreOps(
    const std::unordered_set<pir::Value>& inner_values, ::pir::Operation* op) {
  std::vector<::pir::Operation*> vec_res;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    if (!inner_values.count(op->operand_source(i))) {
      continue;
    }

    vec_res.push_back(op->operand_source(i).defining_op());
  }
  return vec_res;
}

bool CanOpMergeNode(
    const std::unordered_map<::pir::Operation*, GroupClusterNode>& op_path_info,
    ::pir::Operation* pre_op,
    ::pir::Operation* cur_op) {
  const auto& node1 = op_path_info.at(pre_op);
  const auto& node2 = op_path_info.at(cur_op);
  // reduce can not fuse with any op in first stage
  if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*pre_op) ==
      cinn::hlir::framework::kReduction) {
    return false;
  }

  if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*cur_op) ==
      cinn::hlir::framework::kReduction) {
    if (cinn::dialect::ir::GetVectorAttr(cur_op, "dim").size() == 0 ||
        cinn::dialect::ir::GetVectorAttr(cur_op, "dim").size() ==
            cur_op->operand_source(0)
                .type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dims()
                .size()) {
      return false;
    }
  }

  // TODO(phlrain): need update here
  // different loop range can merge, like [128, 128, 1], with [128, 128]
  if ((cinn::hlir::framework::pir::CompatibleInfo::OpKind(*cur_op) !=
       cinn::hlir::framework::kBroadcast) &&
      (op_path_info.at(cur_op).loop_ranges !=
       op_path_info.at(pre_op).loop_ranges)) {
    return false;
  }

  return true;
}

bool ShouldOutputPreNode(
    const std::unordered_map<::pir::Operation*, GroupClusterNode>& op_path_info,
    ::pir::Operation* pre_op,
    ::pir::Operation* cur_op) {
  if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*pre_op) ==
      cinn::hlir::framework::kReduction) {
    return false;
  }

  if (cinn::hlir::framework::pir::CompatibleInfo::OpKind(*cur_op) ==
      cinn::hlir::framework::kReduction) {
    if (cinn::dialect::ir::GetVectorAttr(cur_op, "dim").size() == 0 ||
        cinn::dialect::ir::GetVectorAttr(cur_op, "dim").size() ==
            cur_op->operand_source(0)
                .type()
                .dyn_cast<paddle::dialect::DenseTensorType>()
                .dims()
                .size()) {
      return true;
    }
  }

  // TODO(phlrain): need update here
  // different loop range can merge, like [128, 128, 1], with [128, 128]
  if ((cinn::hlir::framework::pir::CompatibleInfo::OpKind(*cur_op) !=
       cinn::hlir::framework::kBroadcast) &&
      (op_path_info.at(cur_op).loop_ranges !=
       op_path_info.at(pre_op).loop_ranges)) {
    return true;
  }

  return false;
}

std::vector<GroupClusterNode> NodeMergeWithNode(
    const std::vector<GroupClusterNode>& first_stage_output) {
  // stage 2 merge
  // for now we merge node in same pass
  // only for vertical fuse
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
      const auto& node = second_stage_output[node_index];
      const auto& pre_ids = pre_ids_info[node_index];

      GroupClusterNode new_node = node;

      for (auto pre_id : pre_ids) {
        // get pre id

        if (fused_index.count(pre_id)) {
          continue;
        }

        // can new_node merge with pre_id node
        const auto& pre_node = second_stage_output[pre_id];

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
      break;
    }
    second_stage_output.swap(temp_out);
    if (fused == false) {
      break;
    }
  }

  return second_stage_output;
}

std::vector<GroupClusterNode> OpMergeWithOp(cinn::dialect::GroupOp group_op) {
  // op merge with op
  auto inner_values = GetInnerGeneValue(group_op.GetOperators());

  std::unordered_map<::pir::Operation*, GroupClusterNode> op_path;

  auto op_list = group_op.GetOperators();

  std::vector<GroupClusterNode> first_stage_output;

  std::unordered_set<::pir::Operation*> yield_output_ops;
  std::unordered_set<::pir::Operation*> first_output_ops;
  auto yield_op = op_list.back();
  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    if (yield_op->operand_source(i).defining_op()->result(0).use_count() == 1) {
      yield_output_ops.insert(yield_op->operand_source(i).defining_op());
    }
  }

  // first stage op fuse op
  for (auto* op : op_list) {
    if (op->isa<::pir::YieldOp>()) {
      continue;
    }

    auto& cluster_node = op_path[op];
    auto& op_list = cluster_node.ops;

    // process cluster node
    ScheduleInfoNode sch_node;
    GetClusterNodeBasicInfo(op, &cluster_node, &sch_node);

    // process current Node and pre Node
    auto pre_ops = GetPreOps(inner_values, op);
    for (auto pre_op : pre_ops) {
      if (!op_path.count(pre_op)) {
        continue;
      }

      if (CanOpMergeNode(op_path, pre_op, op)) {
        cluster_node.MergePreNode(op_path.at(pre_op), sch_node);
      }

      // TODO(phlrain): should remove this strategy
      if (ShouldOutputPreNode(op_path, pre_op, op)) {
        // Can not merge here, should output pre_op cluster Node
        if (!first_output_ops.count(pre_op)) {
          first_stage_output.push_back(op_path[pre_op]);
          first_output_ops.insert(pre_op);
        }
        continue;
      }
    }

    op_list.push_back(op);

    if (yield_output_ops.count(op) ||
        cinn::hlir::framework::pir::CompatibleInfo::OpKind(*op) ==
            cinn::hlir::framework::kReduction) {
      // TODO(phlrain): yield output no nedd to push into first stage output,
      // Update here
      if (!first_output_ops.count(op)) {
        first_stage_output.push_back(op_path[op]);
        first_output_ops.insert(op);
      }
    }
  }

  return first_stage_output;
}

std::vector<GroupClusterNode> GroupSplit(cinn::dialect::GroupOp group_op) {
  // stage 1
  auto first_stage_output = OpMergeWithOp(group_op);

  if (first_stage_output.size() <= 1) {
    return first_stage_output;
  }

  // stage 2
  auto second_stage_output = NodeMergeWithNode(first_stage_output);

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

std::vector<::pir::Operation*> SortByOriginalOrderAndUniq(
    cinn::dialect::GroupOp group_op,
    const std::vector<::pir::Operation*>& ops) {
  size_t index = 0;
  std::unordered_map<pir::Operation*, size_t> op2order_value;

  for (auto op : group_op.GetOperators()) {
    op2order_value[op] = index++;
  }

  std::vector<pir::Operation*> tmp_ops(ops);
  std::sort(tmp_ops.begin(),
            tmp_ops.end(),
            [&op2order_value](pir::Operation* a, pir::Operation* b) {
              return op2order_value.at(a) < op2order_value.at(b);
            });

  std::unique(tmp_ops.begin(), tmp_ops.end());

  return tmp_ops;
}

std::unordered_map<::pir::Value, size_t> BuildValueOrderByYieldOp(
    const std::vector<GroupClusterNode>& node_list,
    cinn::dialect::GroupOp group_op) {
  std::unordered_map<::pir::Value, size_t> all_output_values;
  auto yield_op = group_op.GetOperators().back();
  for (size_t i = 0; i < yield_op->num_operands(); ++i) {
    size_t id = all_output_values.size();
    all_output_values.emplace(yield_op->operand_source(i), id);
  }

  for (size_t i = 0; i < node_list.size(); ++i) {
    auto node_outside_input = node_list[i].GetOutsideInput();
    for (const auto& val : node_outside_input) {
      size_t id = all_output_values.size();
      all_output_values.emplace(val, id);
    }
  }

  return all_output_values;
}

}  // namespace

class CinnGroupClusterPattern
    : public pir::OpRewritePattern<cinn::dialect::GroupOp> {
 public:
  using pir::OpRewritePattern<cinn::dialect::GroupOp>::OpRewritePattern;

  bool MatchAndRewrite(cinn::dialect::GroupOp group_op,
                       pir::PatternRewriter& rewriter) const override {
    ::pir::IrMapping ir_mapping;

    auto group_outside_input = GetListOutsideInput(group_op.GetOperators());
    // insert initial input to ir mapping
    for (auto val : group_outside_input) {
      ir_mapping.Add(val, val);
    }

    auto split_res = GroupSplit(group_op);

    auto all_output_values = BuildValueOrderByYieldOp(split_res, group_op);

    for (auto& node : split_res) {
      auto output_values = GenerateOutputValue(node.ops, all_output_values);
      auto uniq_ops = SortByOriginalOrderAndUniq(group_op, node.ops);

      auto new_group_op = ReplaceWithGroupOp(
          &rewriter, uniq_ops, node, output_values, &ir_mapping);

      auto& shape_analysis = pir::ShapeAnalysisManager::Instance().Get(
          group_op->GetParentProgram());
      // update ir mapping
      for (size_t i = 0; i < output_values.size(); ++i) {
        ir_mapping.Add(output_values[i], new_group_op->result(i));

        if (shape_analysis.HasShapeOrDataForValue(output_values[i])) {
          shape_analysis.SetShapeOrDataForValue(
              new_group_op->result(i),
              shape_analysis.GetShapeOrDataForValue(output_values[i]));
        }
      }

      for (size_t i = 0; i < output_values.size(); ++i) {
        auto find_it = all_output_values.find(output_values[i]);
        if ((find_it != all_output_values.end()) &&
            (find_it->second < group_op->num_results())) {
          // id < num_results means yield input
          rewriter.ReplaceAllUsesWith(group_op.result(find_it->second),
                                      new_group_op->result(i));
        }
      }
    }
    rewriter.EraseOp(group_op);

    return true;
  }
};

class CinnGroupClusterPass : public pir::PatternRewritePass {
 public:
  CinnGroupClusterPass()
      : pir::PatternRewritePass("cinn_group_cluster_pass", 1) {}

  pir::RewritePatternSet InitializePatterns(pir::IrContext* context) override {
    pir::RewritePatternSet ps(context);

    ps.Add<CinnGroupClusterPattern>(context);

    return ps;
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->num_regions() > 0;
  }
};

std::unique_ptr<pir::Pass> CreateCinnGroupClusterPass() {
  return std::make_unique<CinnGroupClusterPass>();
}

}  // namespace ir
}  // namespace dialect
}  // namespace cinn
