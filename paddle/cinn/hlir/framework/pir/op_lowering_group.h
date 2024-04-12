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

#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "glog/logging.h"

#include "paddle/cinn/common/context.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/framework/pir/utils.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/value.h"
#include "paddle/pir/include/dialect/shape/utils/shape_analysis.h"

namespace cinn {

namespace adt {
class MapExprCtx;
}  // namespace adt

namespace hlir {
namespace framework {
namespace pir {
class OpLoweringGroup {
 public:
  OpLoweringGroup() = default;
  OpLoweringGroup(const OpLoweringGroup&) = delete;
  OpLoweringGroup(OpLoweringGroup&&) = delete;

  explicit OpLoweringGroup(const std::vector<::pir::Operation*>& group_ops)
      : ops_(group_ops) {}

  explicit OpLoweringGroup(std::initializer_list<::pir::Operation*> group_ops)
      : ops_(group_ops) {}

  struct SharedGroupHasher {
    size_t operator()(
        const std::shared_ptr<OpLoweringGroup>& group) const noexcept {
      return std::hash<std::string>()(group->group_id());
    }
  };
  struct SharedGroupComparator {
    bool operator()(
        const std::shared_ptr<OpLoweringGroup>& first,
        const std::shared_ptr<OpLoweringGroup>& second) const noexcept {
      return first->group_id() == second->group_id();
    }
  };

  std::vector<::pir::Value> GetGroupOutputValues() const {
    std::unordered_set<::pir::Operation*> group_ops_set(this->ops_.begin(),
                                                        this->ops_.end());

    std::vector<::pir::Value> output_values;
    for (auto* op : this->ops_) {
      for (size_t i = 0; i < op->num_results(); ++i) {
        auto result = op->result(i);
        if (!result) {
          continue;
        }
        for (auto use_iter = result.use_begin(); use_iter != result.use_end();
             ++use_iter) {
          auto* use_op = use_iter->owner();
          if (group_ops_set.find(use_op) == group_ops_set.end()) {
            output_values.push_back(result);
            break;
          }
        }
      }
    }
    return output_values;
  }

  std::unordered_set<::pir::Value> GetInputOpValues() const {
    std::unordered_set<::pir::Value> group_inputs;

    std::unordered_set<::pir::Operation*> ops_set;
    for (auto op : this->ops_) {
      ops_set.insert(op);
    }

    // count all op's input Value
    for (auto op : this->ops_) {
      for (auto& value : op->operands_source()) {
        if (!value || !value.type()) {
          continue;
        }

        if (!ops_set.count(value.defining_op())) {
          // if the input value owner op is not in OpSet, it's the group's input
          group_inputs.insert(value);
          continue;
        }
      }
    }

    return group_inputs;
  }

  std::unordered_set<::pir::Value> GetOutputOpValues() const {
    std::unordered_set<::pir::Value> group_outputs;

    for (auto op : this->output_ops_) {
      for (auto& result : op->results()) {
        if (!result || result.type()) {
          continue;
        }

        group_outputs.insert(result);
      }
    }
    return group_outputs;
  }

  std::string FuncName() const {
    if (fn_name_ == "") {
      // TODO(Aurelius84): Polish this implementation.
      const_cast<OpLoweringGroup*>(this)->fn_name_ =
          CompatibleInfo::GroupOpsName(ops_);
    }
    return this->fn_name_;
  }

  const symbol::ShapeOrDataDimExprs& GetShapeOrDataExprs(
      const ::pir::Value& value) const {
    CHECK(value_to_shape_or_data_exprs_.count(value))
        << "value not found in value_to_shape_or_data_exprs_";
    return value_to_shape_or_data_exprs_.at(value);
  }

  bool HasShapeOrDataExprs(const ::pir::Value& value) const {
    return value_to_shape_or_data_exprs_.count(value);
  }

  void SetShapeOrDataExprs(const ::pir::Value& value,
                           const symbol::ShapeOrDataDimExprs& shape_or_data) {
    auto iter = value_to_shape_or_data_exprs_.find(value);
    if (iter == value_to_shape_or_data_exprs_.end()) {
      value_to_shape_or_data_exprs_.emplace(value, shape_or_data);
    } else {
      iter->second = shape_or_data;
    }
  }

  void WalkOps(const std::function<void(::pir::Operation*)>& VisitOp) const {
    for (const auto& op : ops_) {
      VisitOp(op);
    }
  }

  const std::vector<::pir::Operation*>& ops() const { return ops_; }

  std::vector<::pir::Operation*>& mut_ops() { return ops_; }

  void SetOps(const std::vector<::pir::Operation*>& new_ops) { ops_ = new_ops; }

  const std::vector<std::string>& input_names() const {
    return this->input_names_;
  }

  std::vector<std::string>& mut_input_names() { return this->input_names_; }

  const std::vector<std::string>& output_names() const {
    return this->output_names_;
  }

  std::vector<std::string>& mut_output_names() { return this->output_names_; }

  const std::vector<::pir::Value>& output_values() const {
    return this->output_values_;
  }

  std::vector<::pir::Value>& mut_output_values() {
    return this->output_values_;
  }

  const std::unordered_set<::pir::Operation*>& output_ops() const {
    return this->output_ops_;
  }

  std::unordered_set<::pir::Operation*>& mut_output_ops() {
    return this->output_ops_;
  }

  std::shared_ptr<adt::MapExprCtx> mut_map_expr_ctx() {
    CHECK_NOTNULL(map_expr_ctx_);
    return map_expr_ctx_;
  }

  const adt::MapExprCtx& map_expr_ctx() const {
    return *CHECK_NOTNULL(map_expr_ctx_);
  }

  void set_value_to_shape_or_data_exprs(
      const std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>&
          value_to_shape_or_data_exprs) {
    value_to_shape_or_data_exprs_ = value_to_shape_or_data_exprs;
  }

  void set_map_expr_ctx(const std::shared_ptr<adt::MapExprCtx>& map_expr_ctx) {
    map_expr_ctx_ = map_expr_ctx;
  }

  const std::string& group_id() const { return this->group_id_; }

  OpPatternKind op_pattern_kind() const { return this->op_pattern_kind_; }

  void set_op_pattern_kind(OpPatternKind pattern_kind) {
    this->op_pattern_kind_ = pattern_kind;
  }

  const std::vector<int64_t>& loop_ranges() const { return loop_ranges_; }

  void set_loop_ranges(const std::vector<int64_t>& loop_ranges) {
    this->loop_ranges_ = loop_ranges;
  }

  const std::vector<symbol::DimExpr>& loop_ranges_expr() const {
    return loop_ranges_expr_;
  }

  void set_loop_ranges_expr(
      const std::vector<symbol::DimExpr>& loop_ranges_expr) {
    this->loop_ranges_expr_ = loop_ranges_expr;
  }

  const std::vector<int64_t>& reduce_axis() const { return reduce_axis_; }

  void set_reduce_axis(const std::vector<int64_t>& reduce_axis) {
    this->reduce_axis_ = reduce_axis;
  }

  const std::map<int, CINNKernelInfo::ArgDimIdx>& int_args_map() const {
    return this->int_args_map_;
  }

  std::map<int, CINNKernelInfo::ArgDimIdx>& mut_int_args_map() {
    return this->int_args_map_;
  }

 private:
  using alignment_schedule_info_t = std::unordered_map<
      ::pir::Operation*,
      std::vector<cinn::hlir::framework::pir::ScheduleInfoNode>>;

 public:
  const alignment_schedule_info_t& alignment_schedule_info() const {
    return alignment_schedule_info_;
  }

  alignment_schedule_info_t& mut_alignment_schedule_info() {
    return alignment_schedule_info_;
  }

  void set_alignment_schedule_info(
      const std::unordered_map<
          ::pir::Operation*,
          std::vector<cinn::hlir::framework::pir::ScheduleInfoNode>>&
          alignment_schedule_info) {
    this->alignment_schedule_info_ = alignment_schedule_info;
  }

  std::shared_ptr<OpLoweringGroup> Clone(::pir::Block* target_block,
                                         ::pir::IrMapping* ir_mapping) const;

 private:
  friend std::ostream& operator<<(std::ostream&, const OpLoweringGroup&);

  // group id, consisted of op's id.
  std::string group_id_{common::UniqName("group_")};
  // op in this group
  std::vector<::pir::Operation*> ops_;
  // output ops of the group.
  std::unordered_set<::pir::Operation*> output_ops_;
  // op pattern kind.
  OpPatternKind op_pattern_kind_{kElementWise};

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<::pir::Value> output_values_;
  std::string fn_name_{""};
  std::map<int, CINNKernelInfo::ArgDimIdx> int_args_map_;

  alignment_schedule_info_t alignment_schedule_info_;
  std::vector<int64_t> reduce_axis_;
  std::vector<int64_t> loop_ranges_;
  std::vector<symbol::DimExpr> loop_ranges_expr_;

  std::shared_ptr<adt::MapExprCtx> map_expr_ctx_;
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_or_data_exprs_;
};

std::ostream& operator<<(std::ostream& os, const OpLoweringGroup& group);
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
