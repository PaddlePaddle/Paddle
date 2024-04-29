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

#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// TODO(SigureMo): Consider use trait to check whether the operation is
// commutative.
static std::unordered_map<std::string, std::vector<std::vector<size_t>>>
    commutative_ops = {
        {"pd_op.multiply", {{0, 1}}},
        {"pd_op.add", {{0, 1}}},
        {"pd_op.maximum", {{0, 1}}},
        {"pd_op.minimum", {{0, 1}}},
        {"pd_op.logical_and", {{0, 1}}},
        {"pd_op.logical_or", {{0, 1}}},
        {"pd_op.logical_xor", {{0, 1}}},
        {"pd_op.equal", {{0, 1}}},
        {"pd_op.not_equal", {{0, 1}}},
        {"pd_op.bitwise_or", {{0, 1}}},
        {"pd_op.bitwise_xor", {{0, 1}}},
        {"pd_op.bitwise_and", {{0, 1}}},
};

template <typename T>
std::vector<T> SortElementsAtIndices(const std::vector<T>& vec,
                                     const std::vector<size_t>& indices) {
  std::vector<T> selected_elements;
  for (auto& idx : indices) {
    selected_elements.push_back(vec[idx]);
  }
  std::sort(selected_elements.begin(), selected_elements.end());
  std::vector<T> sorted_vec;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (std::find(indices.begin(), indices.end(), i) != indices.end()) {
      sorted_vec.push_back(selected_elements.front());
      selected_elements.erase(selected_elements.begin());
    } else {
      sorted_vec.push_back(vec[i]);
    }
  }
  return sorted_vec;
}

std::map<int, int> GetOpInplaceInfo(const pir::Operation* op) {
  std::map<int, int> inplace_info;
  if (!op->HasTrait<paddle::dialect::InplaceTrait>()) {
    return inplace_info;
  }
  pir::IrContext* ctx = pir::IrContext::Instance();
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  }

  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op_name);
  paddle::dialect::OpYamlInfoParser yaml_parser(
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>()
          ->get_op_info_(op_name),
      paddle::dialect::IsLegacyOp(op_name));

  for (size_t i = 0; i < op->num_results(); ++i) {
    std::string value_name = yaml_parser.OutputNames()[i];
    if (yaml_parser.HasInplace(value_name)) {
      const std::string& inplace_name = yaml_parser.InplaceName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(inplace_name);
    }
    if (yaml_parser.HasView(value_name)) {
      const std::string& view_name = yaml_parser.ViewName(value_name);
      inplace_info[i] = yaml_parser.InputName2Id().at(view_name);
    }
  }

  return inplace_info;
}

bool IsTerminateOp(pir::Operation* op) { return op->num_operands() == 0; }
bool IsTerminateValue(const pir::Value& value) {
  return !value.defining_op() || IsTerminateOp(value.defining_op());
}
struct ExpressionTable {
 public:
  ExpressionTable() = default;
  void RegisiterOp(pir::Operation* op) {
    auto op_hash = CalcOperationHash(op);
    auto op_can_be_safe_to_replace = CalcOperationCanBeSafeToReplace(op);
    VLOG(3) << "[RegisiterOp] op " << op->name() << " [" << op << "]"
            << "\n  hash: " << op_hash
            << "\n  can_be_safe_to_replace: " << std::boolalpha
            << op_can_be_safe_to_replace;
    registered_ops_info_[reinterpret_cast<void*>(op)] = {
        op_hash, op_can_be_safe_to_replace};
  }

  size_t GetOperationHash(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        registered_ops_info_.count(reinterpret_cast<void*>(op)),
        1,
        phi::errors::PreconditionNotMet(
            "The operation %s is not registered in the table.", op->name()));
    return registered_ops_info_[reinterpret_cast<void*>(op)].first;
  }

  bool GetOperationCanBeSafeToReplace(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        registered_ops_info_.count(reinterpret_cast<void*>(op)),
        1,
        phi::errors::PreconditionNotMet(
            "The operation %s is not registered in the table.", op->name()));
    return registered_ops_info_[reinterpret_cast<void*>(op)].second;
  }

  void Insert(pir::Operation* op) { common_exprs_[GetOperationHash(op)] = op; }

  std::optional<pir::Operation*> Lookup(pir::Operation* op) {
    VLOG(3) << "[Lookup] op [" << op << "] " << op->name() << " start";
    size_t hash = GetOperationHash(op);
    if (!common_exprs_.count(hash)) {
      return std::nullopt;
    }
    VLOG(3) << "[Lookup] op [" << op << "] " << op->name()
            << " found common subexpression: " << common_exprs_[hash]->name();
    return common_exprs_[hash];
  }

  size_t CalcOperationHash(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        registered_ops_info_.count(reinterpret_cast<void*>(op)),
        0,
        phi::errors::PreconditionNotMet(
            "The operation %s is already registered in the table, don't call "
            "CalcOperationHash twice.",
            op->name()));
    // hash(op) = hash(operands) ^ hash(name) ^ hash(attributes)
    size_t hash = 0;
    VLOG(3) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << " start";

    std::vector<size_t> values_hash;
    for (auto& value : op->operands_source()) {
      values_hash.push_back(CalcValueHash(value));
    }
    if (commutative_ops.count(op->name())) {
      for (auto& commutative_indices : commutative_ops[op->name()]) {
        values_hash = SortElementsAtIndices(values_hash, commutative_indices);
      }
    }
    for (auto& value_hash : values_hash) {
      hash = pir::detail::hash_combine(hash, value_hash);
    }
    VLOG(3) << "[CalcOperationHash] "
            << "value hash: " << hash;
    hash =
        pir::detail::hash_combine(hash, std::hash<std::string>{}(op->name()));
    VLOG(3) << "[CalcOperationHash] "
            << "value + name hash: " << hash;
    for (auto& attr_name : op->info().GetAttributesName()) {
      hash =
          pir::detail::hash_combine(hash, std::hash<std::string>{}(attr_name));
      auto attr = op->attribute(attr_name);
      hash = pir::detail::hash_combine(hash, attr.hash());
    }
    VLOG(3) << "[CalcOperationHash] "
            << "value + name + attr hash: " << hash;
    VLOG(3) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << " hash: " << hash;
    return hash;
  }

  size_t CalcValueHash(const pir::Value& value) {
    // hash(value) = hash(defining_op) ^ value_id
    if (!IsTerminateValue(value)) {
      return pir::detail::hash_combine(GetOperationHash(value.defining_op()),
                                       GetOpResultId(value));
    }
    // hash(termiante_value) = terminate_value_id
    if (!terminate_value_id_map_.count(value)) {
      terminate_value_id_map_[value] = terminate_value_id_++;
    }
    return terminate_value_id_map_[value];
  }

  size_t GetOpResultId(const pir::Value& value) {
    size_t value_id = 0;
    for (auto& result : value.defining_op()->results()) {
      if (value == result) {
        break;
      }
      value_id++;
    }
    return value_id;
  }

  bool CalcOperationCanBeSafeToReplace(pir::Operation* op) {
    PADDLE_ENFORCE_EQ(
        registered_ops_info_.count(reinterpret_cast<void*>(op)),
        0,
        phi::errors::PreconditionNotMet(
            "The operation %s is already registered in the table, don't call "
            "CalcOperationCanBeSafeToReplace twice.",
            op->name()));
    if (op->HasTrait<pir::SideEffectTrait>()) {
      VLOG(3) << "[CalcOperationCanBeSafeToReplace] " << op->name()
              << " has side effect";
      return false;
    }
    for (auto& value : op->operands_source()) {
      if (!IsTerminateValue(value) &&
          !GetOperationCanBeSafeToReplace(value.defining_op())) {
        VLOG(3) << "[CalcOperationCanBeSafeToReplace] " << op->name()
                << " has operand " << value.defining_op()->name()
                << " which can not be safe to replace";
        return false;
      }
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
        auto inplace_info = GetOpInplaceInfo(it->owner());
        for (auto& [out_idx, in_idx] : inplace_info) {
          if (it->owner()->operands()[in_idx].source() == value) {
            VLOG(3) << "[CalcOperationCanBeSafeToReplace] " << op->name()
                    << " has operand " << value.defining_op()->name()
                    << " which is inplace to " << it->owner()->name();
            return false;
          }
        }
      }
    }
    return true;
  }

 private:
  std::unordered_map<size_t, pir::Operation*> common_exprs_;
  std::unordered_map<void*, std::pair<size_t, bool>> registered_ops_info_;
  std::unordered_map<pir::Value, size_t> terminate_value_id_map_;
  size_t terminate_value_id_ = 0;
};

struct CSEAnalyzer {
 public:
  CSEAnalyzer() = default;
  void SimplifyOperation(pir::Operation* op,
                         ExpressionTable* expression_table) {
    VLOG(3) << "[SimplifyOperation] op [" << op << "]";
    if (IsTerminateOp(op)) {
      return;
    }

    expression_table->RegisiterOp(op);
    auto maybe_same_expression = expression_table->Lookup(op);
    if (expression_table->GetOperationCanBeSafeToReplace(op)) {
      if (!maybe_same_expression.has_value()) {
        expression_table->Insert(op);
      } else {
        VLOG(3) << "Found common subexpression: " << op->name();
        to_erase_ops.push_back(
            std::make_pair(op, maybe_same_expression.value()));
      }
    }
    // Handle sub blocks
    for (auto& region : *op) {
      for (auto& block : region) {
        SimplifyBlock(&block, expression_table);
      }
    }
  }

  void SimplifyBlock(pir::Block* block,
                     ExpressionTable* parent_expression_table) {
    // Make a clone to inherit the expressions from parent block
    ExpressionTable expression_table = *parent_expression_table;
    for (auto& op : *block) {
      SimplifyOperation(&op, &expression_table);
    }
  }

  std::vector<std::pair<pir::Operation*, pir::Operation*>> to_erase_ops;
};

pir::Value CreateAssignOp(const pir::Value& value,
                          pir::Operation* op,
                          pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder(ctx, block);
  builder.SetInsertionPointAfter(op);
  auto assign_op = builder.Build<paddle::dialect::AssignOp>(value);
  return assign_op.result(0);
}

void ReplaceOpWith(pir::Operation* op, pir::Operation* new_op) {
  VLOG(3) << "Replacing op " << op->name() << " [" << op << "] with new op "
          << new_op->name() << " [" << new_op << "]";
  PADDLE_ENFORCE_EQ(
      op->num_results(),
      new_op->num_results(),
      phi::errors::InvalidArgument("the num of result should be the same."));
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    auto value = op->result(i);
    auto new_value = new_op->result(i);
    for (auto it = value.use_begin(); it != value.use_end(); ++it) {
      // NOTE(SigureMo): If the value has a shadow output, we could not replace
      // it directly. It will cause a value has two shadow outputs. It is
      // invalid for executor, so we make a copy by inserting a assign op.
      if (it->owner()->isa<pir::ShadowOutputOp>()) {
        new_value = CreateAssignOp(new_value, new_op, op->GetParent());
        break;
      }
    }
    value.ReplaceAllUsesWith(new_value);
  }
  op->Erase();
}

class CommonSubexpressionEliminationPass : public pir::Pass {
 public:
  CommonSubexpressionEliminationPass()
      : pir::Pass("common_subexpression_elimination_pass", 1) {}

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply common_subexpression_elimination_pass";
    int64_t num_erasers{0};
    CSEAnalyzer cse_analyzer;
    ExpressionTable root_expression_table;
    cse_analyzer.SimplifyBlock(op->GetParentProgram()->block(),
                               &root_expression_table);
    VLOG(3) << "Found " << cse_analyzer.to_erase_ops.size()
            << " common subexpression";
    for (auto [op, existing_op] : cse_analyzer.to_erase_ops) {
      VLOG(3) << "Erasing op " << op->name() << " [" << op << "]";
      VLOG(3) << "Replace to op " << existing_op->name() << " [" << existing_op
              << "]";
      ReplaceOpWith(op, existing_op);
      num_erasers++;
    }
    AddStatistics(num_erasers);
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCommonSubexpressionEliminationPass() {
  return std::make_unique<CommonSubexpressionEliminationPass>();
}

}  // namespace pir
