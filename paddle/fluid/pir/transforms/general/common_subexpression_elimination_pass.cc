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

#include "paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"

#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/utils.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

namespace {

// TODO(SigureMo): use compare

using ExpressionMap =
    std::unordered_map<size_t, std::pair<bool, pir::Operation*>>;
using TerminateValueIdMap = std::unordered_map<pir::Value, size_t>;

struct CSEAnalyzer {
 public:
  CSEAnalyzer() = default;

  ExpressionMap expression_map;
  TerminateValueIdMap terminate_value_id_map;
  std::unordered_map<void*, size_t> operation_hash_cache;
  std::vector<pir::Operation*> to_erase_ops;

  size_t value_id = 0;
};

size_t CalcValueHash(const pir::Value& value, CSEAnalyzer* cse_analyzer);
size_t CalcOperationHash(const pir::Operation* op, CSEAnalyzer* cse_analyzer);

bool IsTerminateOp(pir::Operation* op) { return op->num_operands() == 0; }

void SimplifyOperation(pir::Operation* op, CSEAnalyzer* cse_analyzer) {
  if (IsTerminateOp(op)) {
    return;
  }
  // TODO(SigureMo): deal side effect and inplace
  // TODO(SigureMo): deal sub block
  // TODO(SigureMo): Consider reversable operation
  size_t op_hash = CalcOperationHash(op, cse_analyzer);
  if (!cse_analyzer->expression_map.count(op_hash)) {
    cse_analyzer->expression_map[op_hash] = {false, op};
  } else {
    auto& expression = cse_analyzer->expression_map[op_hash];
    VLOG(0) << "Found common subexpression: " << op->name()
            << " , hash: " << op_hash;
    // TODO(SigureMo): deal expression.first
    cse_analyzer->to_erase_ops.push_back(op);
  }
}

void SimplifyBlock(pir::Block* block, CSEAnalyzer* cse_analyzer) {
  for (auto& op : *block) {
    SimplifyOperation(&op, cse_analyzer);
  }
}

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
      if (it->owner()->isa<pir::ShadowOutputOp>()) {
        value = CreateAssignOp(value, it->owner(), op->GetParent());
        break;
      }
    }
    value.ReplaceAllUsesWith(new_value);
  }
  op->Erase();
}

size_t CalcValueHash(const pir::Value& value, CSEAnalyzer* cse_analyzer) {
  // hash(value) = hash(defining_op) ^ value_id
  if (value.defining_op() && !IsTerminateOp(value.defining_op())) {
    size_t value_id = 0;
    for (auto& result : value.defining_op()->results()) {
      if (value == result) {
        break;
      }
      value_id++;
    }
    return pir::detail::hash_combine(
        CalcOperationHash(value.defining_op(), cse_analyzer), value_id);
  }
  // hash(termiante_value) = terminate_value_id
  if (!cse_analyzer->terminate_value_id_map.count(value)) {
    cse_analyzer->terminate_value_id_map[value] = cse_analyzer->value_id++;
  }
  return cse_analyzer->terminate_value_id_map[value];
}

size_t CalcOperationHash(pir::Operation* op, CSEAnalyzer* cse_analyzer) {
  if (cse_analyzer->operation_hash_cache.count(reinterpret_cast<void*>(op))) {
    return cse_analyzer->operation_hash_cache[reinterpret_cast<void*>(op)];
  }
  // hash(op) = hash(operands) ^ hash(name) ^ hash(attributes)
  size_t hash = 0;
  VLOG(0) << "[CalcOperationHash] op [" << op << "] " << op->name() << " start";
  for (auto& value : op->operands_source()) {
    hash = pir::detail::hash_combine(hash, CalcValueHash(value, cse_analyzer));
  }
  VLOG(0) << "[CalcOperationHash] "
          << "value hash: " << hash;
  hash = pir::detail::hash_combine(hash, std::hash<std::string>{}(op->name()));
  VLOG(0) << "[CalcOperationHash] "
          << "value + name hash: " << hash;
  for (auto& attr : op->attributes()) {
    // for (auto& attr_name : op->attributes_num) {
    // TODO(SigureMo): Find the hash function for Attribute, use op_info
    VLOG(0) << "hashing attr name: " << attr.first;
    // TODO(SigureMo): use attributes_name to filter this
    if (attr.first == "op_callstack") {
      continue;
    }
    hash =
        pir::detail::hash_combine(hash, std::hash<std::string>{}(attr.first));
    hash = pir::detail::hash_combine(hash, attr.second.hash());
  }
  VLOG(0) << "[CalcOperationHash] "
          << "value + name + attr hash: " << hash;
  VLOG(0) << "[CalcOperationHash] op [" << op << "] " << op->name()
          << "hash: " << hash;
  cse_analyzer->operation_hash_cache[reinterpret_cast<void*>(op)] = hash;
  return hash;
}

class CommonSubexpressionEliminationPass : public pir::Pass {
 public:
  CommonSubexpressionEliminationPass()
      : pir::Pass("common_subexpression_elimination_pass", 1) {}

  void Run(pir::Operation* op) override {
    VLOG(6) << "apply common_subexpression_elimination_pass";
    int64_t num_erasers{0};
    std::vector<std::string> deleted_vars;
    // bool updated{true};
    // while (updated) {
    //   int64_t pre_num_erasers = num_erasers;
    //   EraseOp(*op->GetParentProgram()->block(), &num_erasers, &deleted_vars);
    //   updated = pre_num_erasers != num_erasers;
    // }
    // if (Has(pir::Pass::kParamScopeAttr)) {
    //   auto scope =
    //   &Get<paddle::framework::Scope>(pir::Pass::kParamScopeAttr); if
    //   (deleted_vars.size() > 0) {
    //     scope->EraseVars(deleted_vars);
    //   }
    // }
    CSEAnalyzer cse_analyzer;
    SimplifyBlock(op->GetParentProgram()->block(), &cse_analyzer);
    for (auto* op : cse_analyzer.to_erase_ops) {
      ReplaceOpWith(
          op,
          cse_analyzer.expression_map[CalcOperationHash(op, &cse_analyzer)]
              .second);
      num_erasers++;
    }
    AddStatistics(num_erasers);
  }

 private:
  void EraseOp(const pir::Block& block,
               int64_t* num_erasers,
               std::vector<std::string>* deleted_vars) {
    std::vector<pir::Operation*> deleted_ops;
    for (auto& op : block) {
      if (op.HasTrait<pir::SideEffectTrait>() ||
          op.isa<paddle::dialect::DataOp>() ||
          paddle::dialect::IsCustomOp(&op)) {
        continue;
      }
      if (op.use_empty()) {
        deleted_ops.push_back(&op);
      }
    }

    for (auto* op : deleted_ops) {
      if (op->isa<pir::ParameterOp>()) {
        auto parameter_op = op->dyn_cast<pir::ParameterOp>();
        deleted_vars->push_back(parameter_op.param_name());
      } else if (op->isa<pir::ConstantTensorOp>()) {
        auto constant_tensor_op = op->dyn_cast<pir::ConstantTensorOp>();
        deleted_vars->push_back(constant_tensor_op.tensor_name());
      }
      op->Erase();
      (*num_erasers)++;
    }

    if (deleted_ops.empty()) {
      for (auto& op : block) {
        for (size_t i = 0; i < op.num_regions(); ++i) {
          auto& inner_region = op.region(i);
          for (auto& inner_block : inner_region) {
            EraseOp(inner_block, num_erasers, deleted_vars);
          }
        }
      }
    } else {
      EraseOp(block, num_erasers, deleted_vars);
    }
  }
};

}  // namespace

namespace pir {

std::unique_ptr<Pass> CreateCommonSubexpressionEliminationPass() {
  return std::make_unique<CommonSubexpressionEliminationPass>();
}

}  // namespace pir
