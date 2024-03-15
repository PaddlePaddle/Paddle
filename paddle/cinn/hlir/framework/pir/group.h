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
using framework::OpPatternKind;

struct Group {
  // Control the clone strategy for Group.
  class Options {
   public:
    Options() : only_clone_ops(true) {}
    bool OnlyCloneOps() const { return only_clone_ops; }

   private:
    bool only_clone_ops = false;
  };

 public:
  Group() = default;
  Group(const Group&) = delete;
  Group(Group&&) = delete;

  explicit Group(const std::vector<::pir::Operation*>& group_ops)
      : ops(group_ops) {}

  explicit Group(std::initializer_list<::pir::Operation*> group_ops)
      : ops(group_ops) {}

  std::shared_ptr<Group> Clone(::pir::Block* target_block,
                               ::pir::IrMapping& ir_mapping,
                               const Options& option = Options()) const;

  const symbol::ShapeOrDataDimExprs& GetShapeOrDataExprs(
      const ::pir::Value& value) const {
    CHECK(value_to_shape_or_data_exprs_.count(value))
        << "value not found in value_to_shape_or_data_exprs_";
    return value_to_shape_or_data_exprs_.at(value);
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

  void set_value_to_shape_or_data_exprs(
      const std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>&
          value_to_shape_or_data_exprs) {
    value_to_shape_or_data_exprs_ = value_to_shape_or_data_exprs;
  }

  // distance to last group.
  int depth{0};
  int max_depth{0};
  int min_depth{INT_MAX};
  // group id, consisted of op's id.
  std::string group_id{""};
  // global unique id.
  std::string unique_id{"uniq"};
  // op in this group
  std::vector<::pir::Operation*> ops;
  std::unordered_set<::pir::Operation*> ops_set;
  // input ops of the group.
  std::unordered_map<::pir::Operation*, int> input_ops;
  // output ops of the group.
  std::unordered_set<::pir::Operation*> output_ops;
  // op pattern kind.
  OpPatternKind op_pattern_kind{kElementWise};
  // internal op, the output is used by multi-op.
  // internal op can't use compute inline, should use buffer.
  std::unordered_set<::pir::Operation*> internal_ops;
  // master op for schedule
  std::unordered_set<::pir::Operation*> master_ops;

  // fused sub-groups, used for fusion merge pass
  std::vector<std::shared_ptr<Group>> fused_sub_groups;
  // if as sub-group, used for belong groups.
  std::unordered_set<std::shared_ptr<Group>> belong_groups;

  // for op lowering.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<::pir::Value> output_values;
  std::string fn_name{""};
  std::map<int, CINNKernelInfo::ArgDimIdx> int_args_map;

  std::unordered_map<::pir::Operation*,
                     std::vector<cinn::hlir::framework::pir::ScheduleInfoNode>>
      alignment_schedule_info;
  std::vector<int64_t> reduce_axis;
  std::vector<int64_t> loop_ranges;
  std::vector<symbol::DimExpr> loop_ranges_expr;

  struct SharedGroupHasher {
    size_t operator()(const std::shared_ptr<Group>& group) const noexcept {
      return std::hash<uint64_t>()(reinterpret_cast<uint64_t>(group.get()));
    }
  };
  struct SharedGroupComparator {
    bool operator()(const std::shared_ptr<Group>& first,
                    const std::shared_ptr<Group>& second) const noexcept {
      return first.get() == second.get();
    }
  };

  std::vector<::pir::Operation*> CollectOps() const {
    if (fused_sub_groups.size()) {
      std::vector<::pir::Operation*> tmp_ops;
      for (auto& group : fused_sub_groups) {
        tmp_ops.insert(tmp_ops.end(), group->ops.begin(), group->ops.end());
      }
      return tmp_ops;
    } else {
      return ops;
    }
  }

  void WalkOps(const std::function<void(::pir::Operation*)>& VisitOp) const {
    if (fused_sub_groups.size()) {
      for (auto& group : fused_sub_groups) {
        for (const auto& op : group->ops) {
          VisitOp(op);
        }
      }
    } else {
      for (const auto& op : ops) {
        VisitOp(op);
      }
    }
  }

  std::unordered_set<::pir::Operation*> OpSet() const {
    std::unordered_set<::pir::Operation*> op_set;
    for (auto op : CollectOps()) {
      op_set.insert(op);
    }
    return op_set;
  }

  std::unordered_set<::pir::Value> GetInputOpValues() const {
    std::unordered_set<::pir::Value> group_inputs;
    auto ops_set = this->OpSet();
    // count all op's input Value
    for (auto op : this->CollectOps()) {
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

    for (auto op : this->output_ops) {
      for (auto& result : op->results()) {
        if (!result || result.type()) {
          continue;
        }

        group_outputs.insert(result);
      }
    }
    return group_outputs;
  }

  const std::vector<::pir::Value>& GetGroupOutputValues() const {
    return this->output_values;
  }

  std::string GetFuncName() { return "fn_" + group_id + unique_id; }

  std::vector<::pir::Value> GenerateGroupOutputValues() const {
    std::unordered_set<::pir::Operation*> group_ops_set(this->ops.begin(),
                                                        this->ops.end());

    std::vector<::pir::Value> output_values;
    for (auto* op : this->ops) {
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

  std::shared_ptr<adt::MapExprCtx> mut_map_expr_ctx() {
    CHECK_NOTNULL(map_expr_ctx_);
    return map_expr_ctx_;
  }

  const adt::MapExprCtx& map_expr_ctx() const {
    return *CHECK_NOTNULL(map_expr_ctx_);
  }

  void set_map_expr_ctx(const std::shared_ptr<adt::MapExprCtx>& map_expr_ctx) {
    map_expr_ctx_ = map_expr_ctx;
  }

 public:
  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  producer_groups() const {
    return producer_groups_;
  }

  const std::unordered_set<std::shared_ptr<Group>,
                           SharedGroupHasher,
                           SharedGroupComparator>&
  consumer_groups() const {
    return consumer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_producer_groups() {
    return &producer_groups_;
  }

  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>*
  mut_consumer_groups() {
    return &consumer_groups_;
  }

  OpPatternKind kind() const { return op_pattern_kind; }

  std::string FuncName() const {
    if (fn_name == "") {
      // TODO(Aurelius84): Polish this implementation.
      const_cast<Group*>(this)->fn_name = CompatibleInfo::GroupOpsName(ops);
    }
    return this->fn_name;
  }

 private:
  // input groups
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      producer_groups_;
  // output grous
  std::unordered_set<std::shared_ptr<Group>,
                     SharedGroupHasher,
                     SharedGroupComparator>
      consumer_groups_;
  std::shared_ptr<adt::MapExprCtx> map_expr_ctx_;

  std::unordered_map<::pir::Value, symbol::ShapeOrDataDimExprs>
      value_to_shape_or_data_exprs_;
};

std::ostream& operator<<(std::ostream& os, const Group& group);

}  // namespace pir
}  // namespace framework
}  // namespace hlir
}  // namespace cinn
