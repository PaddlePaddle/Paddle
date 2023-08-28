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

#include "paddle/fluid/ir/dialect/paddle_dialect/utils/op_yaml_info_parser.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/pass/pass.h"

namespace paddle {
namespace dialect {

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

std::unordered_map<ir::Operation*, std::unordered_set<ir::Value>>
GetEagerDeletionValues(ir::Block* block) {
  std::unordered_map<ir::Value, ir::Operation*> value_2_op;
  for (auto& op : *block) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto input_value = op->operand_source(i);
      if (!input_value || !ValueCanBeDeleted(input_value)) {
        VLOG(0) << "The " << i << "-th input value of the Operation("
                << op->name() << ") can not be deleted.";
        continue;
      }
      if (IsNoNeedBufferValue(op, input_value)) {
        VLOG(0) << "The " << i << "-th input value of the Operation("
                << op->name() << ") is no need buffer, so can not be deleted.";
        continue;
      }
      value_2_op[input_value] = op;
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
  const auto eager_deletion_input_values = GetEagerDeletionInputValues(block);

  std::unordered_map<ir::Operation*, ir::OpInfo> inplace_ops;
  for (auto& op : *block) {
    if (op->HasTrait<paddle::dialect::InplaceTrait>()) {
      VLOG(0) << "Operation " << op->name()
              << " doesn't have inplace version." continue;
    }

    std::string inplace_op_name = op->name() + "_";
    ir::OpInfo inplace_op_info =
        ir::IrContext::Instance()->GetRegisteredOpInfo(inplace_op_name);
    if (!inplace_op_info) {
      VLOG(0) << "Operation " << op->name()
              << " doesn't have inplace version." continue;
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

    bool can_do_inplace = true;
    for (auto& kv : inplace_out_2_in) {
      if (eager_deletion_input_values.at(op).count(
              op->operand_source(kv.second)) == 0) {
        can_do_inplace = false;
      }
    }
    if (can_do_inplace) {
      inplace_ops[op] = inplace_op_info;
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

      ir::Block::iterator insert_pos =
          std::find(block->begin(), block->end(), origin_op);

      PADDLE_ENFORCE_NE(
          insert_pos,
          block->end(),
          paddle::platform::errors::NotFound("Operator %s not found in block.",
                                             origin_op->name()));

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

std::unique_ptr<Pass> CreateInplacePass() {
  return std::make_unique<InplacePass>();
}

}  // namespace dialect
}  // namespace paddle
