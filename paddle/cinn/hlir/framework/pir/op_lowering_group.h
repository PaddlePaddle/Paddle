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
#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/common/enforce.h"
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
  OpLoweringGroup(const OpLoweringGroup&) = delete;
  OpLoweringGroup(OpLoweringGroup&&) = delete;

  explicit OpLoweringGroup(const std::vector<::pir::Operation*>& group_ops)
      : ops_(group_ops) {
    fn_name_ = CompatibleInfo::GroupOpsName(ops_);
  }

  explicit OpLoweringGroup(std::initializer_list<::pir::Operation*> group_ops)
      : ops_(group_ops) {
    fn_name_ = CompatibleInfo::GroupOpsName(ops_);
  }

  const std::string& FuncName() const { return this->fn_name_; }
  ::pir::Block* GetParentBlock() const;
  ::pir::Program* GetParentProgram() const;
  std::vector<::pir::Value> GetGroupOutputValues() const;
  std::vector<::pir::Value> GetInputOpValues() const;
  std::unordered_set<::pir::Value> GetOutputOpValues() const;
  const symbol::ShapeOrDataDimExprs& GetShapeOrDataExprs(
      const ::pir::Value& value) const;

  bool HasShapeOrDataExprs(const ::pir::Value& value) const {
    return value_to_shape_or_data_exprs_.count(value);
  }

  void SetShapeOrDataExprs(const ::pir::Value& value,
                           const symbol::ShapeOrDataDimExprs& shape_or_data);

  void WalkOps(const std::function<void(::pir::Operation*)>& VisitOp) const {
    for (const auto& op : ops_) {
      VisitOp(op);
    }
  }

  bool IsBroadcastLeaf() const { return is_broadcast_leaf_; }
  void SetIsBroadcastLeaf(bool is_broadcast_leaf) {
    is_broadcast_leaf_ = is_broadcast_leaf;
  }

  const std::vector<::pir::Operation*>& ops() const { return ops_; }
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
    PADDLE_ENFORCE_NOT_NULL(
        map_expr_ctx_,
        ::common::errors::Unavailable("Required map_expr_ctx_ != nullptr."));
    return map_expr_ctx_;
  }

  const adt::MapExprCtx& map_expr_ctx() const {
    PADDLE_ENFORCE_NOT_NULL(
        map_expr_ctx_,
        ::common::errors::Unavailable("Required map_expr_ctx_ != nullptr."));
    return *map_expr_ctx_;
  }

  void set_value_to_shape_or_data_exprs(
      const std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>&
          value_to_shape_or_data_exprs) {
    value_to_shape_or_data_exprs_ = value_to_shape_or_data_exprs;
  }

  void set_map_expr_ctx(const std::shared_ptr<adt::MapExprCtx>& map_expr_ctx) {
    map_expr_ctx_ = map_expr_ctx;
  }

  void set_output_values_and_ops() {
    auto yield_op = ops_.back();
    for (size_t i = 0; i < yield_op->num_operands(); ++i) {
      auto in = yield_op->operand_source(i);
      output_values_.push_back(in);
      output_ops_.insert(in.defining_op());
    }
  }

  const std::string& group_id() const { return this->group_id_; }

  OpPatternKind op_pattern_kind() const { return this->op_pattern_kind_; }

  void set_op_pattern_kind(OpPatternKind pattern_kind) {
    this->op_pattern_kind_ = pattern_kind;
  }

  void set_backend_optim_info(const cinn::dialect::GroupInfo& attr) {
    this->backend_optim_info_.emplace(attr.alignment_schedule_info,attr.reduce_axis,attr.loop_ranges,attr.loop_ranges_expr);
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
      struct BackendOptimInfo {
    alignment_schedule_info_t alignment_schedule_info_;
    std::vector<int64_t> reduce_axis_;
    std::vector<int64_t> loop_ranges_;
    std::vector<symbol::DimExpr> loop_ranges_expr_;
    explicit BackendOptimInfo(const alignment_schedule_info_t& alignment_schedule_info,const std::vector<int64_t>& reduce_axis,const std::vector<int64_t>& loop_ranges,const std::vector<symbol::DimExpr>& loop_ranges_expr) :
    alignment_schedule_info_(alignment_schedule_info),reduce_axis_(reduce_axis),loop_ranges_(loop_ranges),loop_ranges_expr_(loop_ranges_expr){}
    //BackendOptimInfo(const BackendOptimInfo&) = delete;
    //BackendOptimInfo(BackendOptimInfo&&) = delete;
  };

 public:

  alignment_schedule_info_t& mut_alignment_schedule_info() {
    return backend_optim_info_->alignment_schedule_info_;
  }

    const std::optional<BackendOptimInfo>& backend_optim_info() const {
    return this->backend_optim_info_;
  }

  std::shared_ptr<OpLoweringGroup> Clone(::pir::Block* target_block,
                                         ::pir::IrMapping* ir_mapping) const;

 private:

  friend std::ostream& operator<<(std::ostream&, const OpLoweringGroup&);

  // group id, consisted of op's id.
  std::string group_id_{common::UniqName("group_")};
  // op in this group
  std::vector<::pir::Operation*> ops_;
  std::string fn_name_;
  // output ops of the group.
  std::unordered_set<::pir::Operation*> output_ops_;
  // op pattern kind.
  OpPatternKind op_pattern_kind_{kElementWise};
  // FIXME(Aurelius84): Need more elegent way to deal with SymDimExprs
  // from local and global ShapeAnalysis. It will be removed after
  // refactoring logic of OpLoweringGroup.
  bool is_broadcast_leaf_{false};

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<::pir::Value> output_values_;
  std::map<int, CINNKernelInfo::ArgDimIdx> int_args_map_;

  std::optional<BackendOptimInfo> backend_optim_info_;

  std::shared_ptr<adt::MapExprCtx> map_expr_ctx_;
  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_or_data_exprs_;
};

std::ostream& operator<<(std::ostream& os, const OpLoweringGroup& group);
}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
