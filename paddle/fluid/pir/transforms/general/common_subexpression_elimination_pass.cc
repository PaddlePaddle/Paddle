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
std::vector<T> SortIndicesElements(const std::vector<T>& vec,
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
  void Insert(pir::Operation* op) { known_ops_[CalcOperationHash(op)] = op; }

  std::optional<pir::Operation*> Lookup(pir::Operation* op) {
    VLOG(6) << "[Lookup] op [" << op << "] " << op->name() << " start";
    size_t hash = CalcOperationHash(op);
    if (!known_ops_.count(hash)) {
      return std::nullopt;
    }
    VLOG(6) << "[Lookup] op [" << op << "] " << op->name()
            << " found common subexpression: " << known_ops_[hash]->name();
    return known_ops_[hash];
  }

  size_t CalcOperationHash(pir::Operation* op) {
    if (operation_hash_cache_.count(reinterpret_cast<void*>(op))) {
      return operation_hash_cache_[reinterpret_cast<void*>(op)];
    }
    // hash(op) = hash(operands) ^ hash(name) ^ hash(attributes)
    size_t hash = 0;
    VLOG(6) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << " start";

    std::vector<size_t> values_hash;
    for (auto& value : op->operands_source()) {
      // hash = pir::detail::hash_combine(hash, CalcValueHash(value));
      values_hash.push_back(CalcValueHash(value));
    }
    if (commutative_ops.count(op->name())) {
      for (auto& commutative_indices : commutative_ops[op->name()]) {
        values_hash = SortIndicesElements(values_hash, commutative_indices);
      }
    }
    for (auto& value_hash : values_hash) {
      hash = pir::detail::hash_combine(hash, value_hash);
    }
    VLOG(6) << "[CalcOperationHash] "
            << "value hash: " << hash;
    hash =
        pir::detail::hash_combine(hash, std::hash<std::string>{}(op->name()));
    VLOG(6) << "[CalcOperationHash] "
            << "value + name hash: " << hash;
    for (auto& attr_name : op->info().GetAttributesName()) {
      hash =
          pir::detail::hash_combine(hash, std::hash<std::string>{}(attr_name));
      auto attr = op->attribute(attr_name);
      hash = pir::detail::hash_combine(hash, attr.hash());
    }
    VLOG(6) << "[CalcOperationHash] "
            << "value + name + attr hash: " << hash;
    VLOG(6) << "[CalcOperationHash] op [" << op << "] " << op->name()
            << "hash: " << hash;
    operation_hash_cache_[reinterpret_cast<void*>(op)] = hash;
    return hash;
  }

  size_t CalcValueHash(const pir::Value& value) {
    // hash(value) = hash(defining_op) ^ value_id
    if (!IsTerminateValue(value)) {
      return pir::detail::hash_combine(CalcOperationHash(value.defining_op()),
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

 private:
  std::unordered_map<size_t, pir::Operation*> known_ops_;
  std::unordered_map<void*, size_t> operation_hash_cache_;
  std::unordered_map<pir::Value, size_t> terminate_value_id_map_;
  size_t terminate_value_id_ = 0;
};

using TerminateValueIdMap = std::unordered_map<pir::Value, size_t>;

struct CSEAnalyzer {
 public:
  CSEAnalyzer() = default;
  void SimplifyOperation(pir::Operation* op) {
    VLOG(6) << "[SimplifyOperation] op [" << op << "]";
    if (IsTerminateOp(op)) {
      return;
    }
    auto maybe_same_expression = expression_table.Lookup(op);
    if (!maybe_same_expression.has_value()) {
      expression_table.Insert(op);
    } else {
      VLOG(6) << "Found common subexpression: " << op->name();
      to_erase_ops.push_back(op);
    }
    // Handle sub blocks
    for (auto& region : *op) {
      for (auto& block : region) {
        SimplifyBlock(&block);
      }
    }
  }

  void SimplifyBlock(pir::Block* block) {
    for (auto& op : *block) {
      SimplifyOperation(&op);
    }
  }

  bool CanBeSafeToReplace(pir::Operation* op) {
    if (op->HasTrait<pir::SideEffectTrait>()) {
      return false;
    }
    for (auto& value : op->operands_source()) {
      if (!CanBeSafeToReplace(value.defining_op())) {
        return false;
      }
      for (auto it = value.use_begin(); it != value.use_end(); ++it) {
        auto inplace_info = GetOpInplaceInfo(it->owner());
        for (auto& [out_idx, in_idx] : inplace_info) {
          if (it->owner()->operands()[in_idx].source() == value) {
            return false;
          }
        }
      }
    }
    return true;
  }

  ExpressionTable expression_table;
  std::vector<pir::Operation*> to_erase_ops;
};

pir::Value CreateAssignOp(const pir::Value& value,
                          pir::Operation* op,
                          pir::Block* block) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Builder builder(ctx, block);
  builder.set_insertion_point(op);
  builder.Build<paddle::dialect::AssignOp>(value);
  return value;
}

void ReplaceOpWith(pir::Operation* op, pir::Operation* new_op) {
  VLOG(0) << "Replacing op " << op->name() << " [" << op << "] with new op "
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
        value = CreateAssignOp(value, it->owner(), op->GetParent());
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
    cse_analyzer.SimplifyBlock(op->GetParentProgram()->block());
    for (auto* op : cse_analyzer.to_erase_ops) {
      auto existing_op = cse_analyzer.expression_table.Lookup(op).value();
      if (!cse_analyzer.CanBeSafeToReplace(existing_op) ||
          !cse_analyzer.CanBeSafeToReplace(op)) {
        continue;
      }
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
