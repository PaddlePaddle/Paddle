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

#include "paddle/fluid/ir/transforms/inplace_pass.h"

#include "paddle/fluid/ir/dialect/paddle_dialect/interface/op_yaml_info.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/trait/inplace.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_parser.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_registry.h"

bool ValueCanBeDeleted(ir::Value value) {
  bool is_persisable = false;
  auto* defined_op = value.GetDefiningOp();
  if (defined_op->HasAttribute(kAttrIsPersisable)) {
    ir::OpResult result = value.dyn_cast<::ir::OpResult>();
    is_persisable = defined_op->attribute(kAttrIsPersisable)
                        .dyn_cast<::ir::ArrayAttribute>()
                        .AsVector()[result.GetResultIndex()]
                        .dyn_cast<::ir::BoolAttribute>()
                        .data();
  }
  if (is_persisable) {
    return false;
  }

  return value.type().isa<paddle::dialect::DenseTensorType>() ||
         value.type().isa<paddle::dialect::SelectedRowsType>();
}

bool ValueCanDoInplace(ir::Type input_type, ir::Type output_type) {
  if (input_type != output_type) {
    return false;
  }
  return true;
}

bool IsNoNeedBufferValue(ir::Operation* op, ir::Value value) {
  ir::OpInfo op_info = op->info();
  auto yaml_interface =
      op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
  if (yaml_interface) {
    paddle::dialect::OpYamlInfoParser yaml_info_parser(
        yaml_interface->get_op_info_());
    auto& no_need_buffer_ids = yaml_info_parser.NoNeedBufferIds();
    for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
      if (value == op->operand_source(no_need_buffer_ids[id])) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_set<ir::Value> GetSkipDeletionValues(ir::Block* block) {
  std::unordered_set<ir::Value> skip_gc_values;
  // NOTE(zhangbo): pd.feed's output and pd.fetch's input can not be eager
  // deleted.
  for (auto& op : *block) {
    if (op->name() == "pd.feed" || op->name() == "pd.data") {
      skip_gc_values.insert(op->result(0));
      continue;
    }
    if (op->name() == "pd.fetch" || op->name() == "pd.shadow_output") {
      skip_gc_values.insert(op->operand_source(0));
      continue;
    }
  }
  return skip_gc_values;
}

std::unordered_map<ir::Operation*, std::unordered_set<ir::Value>>
GetEagerDeletionValues(ir::Block* block) {
  std::unordered_set<ir::Value> skip_deletion_values =
      GetSkipDeletionValues(block);
  std::unordered_map<ir::Value, ir::Operation*> value_2_op;
  for (auto& op : *block) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto input_value = op->operand_source(i);
      if (skip_deletion_values.count(input_value) > 0) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << op->name() << ") can not be deleted.";
        continue;
      }
      if (!input_value || !ValueCanBeDeleted(input_value)) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << op->name() << ") can not be deleted.";
        continue;
      }
      if (IsNoNeedBufferValue(op, input_value)) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << op->name() << ") is no need buffer, so can not be deleted.";
        continue;
      }
      value_2_op[input_value] = op;
    }

    for (size_t i = 0; i < op->num_results(); ++i) {
      ir::Value output_value = op->result(i);
      if (ValueCanBeDeleted(output_value)) {
        value_2_op[output_value] = op;
      }
    }
  }

  std::unordered_map<ir::Operation*, std::unordered_set<ir::Value>>
      eager_deletion_values;
  for (auto& value_op_pair : value_2_op) {
    ir::Value value = value_op_pair.first;
    ir::Operation* op = value_op_pair.second;
    eager_deletion_values[op].insert(value);
  }

  return eager_deletion_values;
}

std::unordered_map<ir::Operation*, ir::OpInfo> GetInplaceOp(ir::Block* block) {
  const auto eager_deletion_input_values = GetEagerDeletionValues(block);

  std::unordered_set<ir::Value> visited_values;
  std::unordered_set<ir::Value> reused_input_values;
  std::unordered_set<ir::Value> reused_output_values;

  std::unordered_map<ir::Operation*, ir::OpInfo> inplace_ops;
  for (auto& op : *block) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      visited_values.insert(op->operand_source(i));
    }

    if (eager_deletion_input_values.count(op) == 0) {
      VLOG(6)
          << "Operation " << op->name()
          << " not in eager_deletion_input_values, so that can not do inplace.";
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    if (op->HasTrait<paddle::dialect::InplaceTrait>()) {
      VLOG(6) << "Operation " << op->name() << " is already an inplace op.";
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }

      for (size_t i = 0; i < op->num_operands(); ++i) {
        reused_input_values.insert(op->operand_source(i));
      }
      for (size_t i = 0; i < op->num_results(); ++i) {
        reused_output_values.insert(op->result(i));
      }

      continue;
    }

    std::string inplace_op_name = op->name() + "_";
    ir::OpInfo inplace_op_info =
        ir::IrContext::Instance()->GetRegisteredOpInfo(inplace_op_name);
    if (!inplace_op_info) {
      VLOG(6) << "Operation " << op->name()
              << " doesn't have inplace version, so that can not do inplace.";
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    auto inplace_op_yaml_interface =
        inplace_op_info
            .GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    PADDLE_ENFORCE_NOT_NULL(
        inplace_op_yaml_interface,
        phi::errors::PreconditionNotMet(
            "can not find OpYamlInfoInterface from [%s]", op->name() + "_"));
    paddle::dialect::OpYamlInfoParser inplace_op_yaml_info_parser(
        inplace_op_yaml_interface->get_op_info_());
    std::unordered_map<int, int> inplace_out_2_in =
        inplace_op_yaml_info_parser.GetInplaceIdMap();

    std::unordered_set<ir::Value> op_inputs;
    for (size_t i = 0; i < op->num_operands(); ++i) {
      op_inputs.insert(op->operand_source(i));
    }
    std::unordered_set<ir::Value> op_outputs;
    for (size_t i = 0; i < op->num_results(); ++i) {
      op_outputs.insert(op->result(i));
    }
    std::unordered_set<ir::Value> valid_values;
    for (const auto& value : eager_deletion_input_values.at(op)) {
      if (value && op_inputs.count(value) != 0 &&
          op_outputs.count(value) == 0) {
        valid_values.insert(value);
      }
    }

    if (valid_values.empty()) {
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    bool can_do_inplace = true;
    for (auto& kv : inplace_out_2_in) {
      int out_slot = kv.first;
      int in_slot = kv.second;
      if (!ValueCanDoInplace(op->operand_source(in_slot).type(),
                             op->result(out_slot).type())) {
        can_do_inplace = false;
        break;
      }

      if (valid_values.count(op->operand_source(in_slot)) == 0) {
        can_do_inplace = false;
        break;
      }

      if (visited_values.count(op->result(out_slot)) > 0) {
        can_do_inplace = false;
        break;
      }

      if (!ValueCanBeDeleted(op->result(out_slot))) {
        can_do_inplace = false;
        break;
      }

      if (reused_input_values.count(op->operand_source(in_slot)) > 0 ||
          reused_output_values.count(op->result(out_slot)) > 0) {
        can_do_inplace = false;
        break;
      }
    }

    if (can_do_inplace) {
      inplace_ops[op] = inplace_op_info;
      VLOG(6) << op->name() << " has inplace version op: " << inplace_op_name;
      for (auto& kv : inplace_out_2_in) {
        reused_input_values.insert(op->operand_source(kv.second));
        reused_output_values.insert(op->result(kv.first));
      }
    }

    for (size_t i = 0; i < op->num_results(); ++i) {
      visited_values.insert(op->result(i));
    }
  }
  return inplace_ops;
}

class InplacePass : public ir::Pass {
 public:
  InplacePass() : ir::Pass("InplacePass", 3) {}

  void Run(ir::Operation* op) override {
    auto module_op = op->dyn_cast<ir::ModuleOp>();
    IR_ENFORCE(module_op, "DcePass should run on module op.");
    auto* block = module_op.block();
    auto inplace_ops = GetInplaceOp(block);

    for (auto kv : inplace_ops) {
      ir::Operation* origin_op = kv.first;
      VLOG(6) << "Do inplace for: " << origin_op->name();

      ir::Block::iterator insert_pos =
          std::find(block->begin(), block->end(), origin_op);

      IR_ENFORCE(insert_pos != block->end(),
                 "Operator %s not found in block.",
                 origin_op->name());

      std::vector<ir::OpResult> inputs;
      for (size_t i = 0; i < origin_op->num_operands(); ++i) {
        inputs.emplace_back(
            origin_op->operand_source(i).dyn_cast<::ir::OpResult>());
      }

      std::vector<ir::Type> outputs;
      for (size_t i = 0; i < origin_op->num_results(); ++i) {
        outputs.emplace_back(origin_op->result(i).type());
      }

      ir::Operation* inplace_op = ir::Operation::Create(
          inputs, origin_op->attributes(), outputs, kv.second);

      for (size_t i = 0; i < origin_op->num_results(); ++i) {
        origin_op->result(i).ReplaceAllUsesWith(inplace_op->result(i));
      }

      block->insert(insert_pos, inplace_op);
      block->erase(insert_pos);
    }
  }

  bool CanApplyOn(ir::Operation* op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }
};

namespace ir {

std::unique_ptr<ir::Pass> CreateInplacePass() {
  return std::make_unique<InplacePass>();
}

}  // namespace ir

REGISTER_PASS(inplace, InplacePass);
