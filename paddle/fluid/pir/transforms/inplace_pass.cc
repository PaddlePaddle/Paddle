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

#include "paddle/fluid/pir/transforms/inplace_pass.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

namespace details {
// NOTE(zhangbo): Which kind of value can be deleted?
// (1) Value's type needs to be AllocatedDenseTensorType or
// AllocatedSelectedRowsType; (2) Value's is not persisable.
static bool CanBeDeleted(pir::Value value) {
  if (!value.type()) {
    return false;
  }
  if (!value.type().isa<paddle::dialect::AllocatedDenseTensorType>() &&
      !value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    return false;
  }
  if (value.GetDefiningOp()->HasAttribute(kAttrIsPersisable)) {
    return !(value.GetDefiningOp()
                 ->attribute(kAttrIsPersisable)
                 .dyn_cast<pir::ArrayAttribute>()
                 .AsVector()[value.dyn_cast<pir::OpResult>().GetResultIndex()]
                 .dyn_cast<pir::BoolAttribute>()
                 .data());
  }
  return true;
}

static bool CanDoInplace(const std::unordered_set<pir::Value>& eager_dels,
                         pir::Value input,
                         pir::Value output) {
  if (input.type() != output.type()) {
    VLOG(9) << "     -- input's type != output's type, can't do inplace";
    return false;
  }
  if (eager_dels.count(input) == 0) {
    VLOG(9) << "     -- input not in eager_deletion_valus, can't do inplace";
    return false;
  }
  return true;
}

static bool IsNoNeedBuffer(pir::Operation* op, pir::Value value) {
  if (op->dialect()->name().compare(paddle::dialect::KernelDialect::name()) !=
      0) {
    VLOG(8) << op->name()
            << "is not a kernel_dialect op, no need buffer is false";
    return false;
  }
  auto op_name =
      op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();
  pir::OpInfo op_info =
      pir::IrContext::Instance()->GetRegisteredOpInfo(op_name);
  if (op_info) {
    auto info_interface =
        op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    if (info_interface) {
      paddle::dialect::OpYamlInfoParser info_parser(
          info_interface->get_op_info_());
      auto& no_need_buffer_ids = info_parser.NoNeedBufferIds();
      for (size_t id = 0; id < no_need_buffer_ids.size(); id++) {
        if (value == op->operand_source(no_need_buffer_ids[id])) {
          return true;
        }
      }
    }
  }
  return false;
}

// NOTE(zhangbo): pd_op.feed's output and pd_op.fetch's input can not be eager
// deleted.
static std::unordered_set<pir::Value> GetSkipDeletionValues(pir::Block* block) {
  std::unordered_set<pir::Value> skip_dels;
  for (auto& op : *block) {
    if (op->dialect()->name().compare(paddle::dialect::KernelDialect::name()) !=
        0) {
      continue;
    }
    IR_ENFORCE(op->attributes().count("op_name") > 0,
               "kernel_dialect op should own an 'op_name' attribute.");
    auto upper_op_name =
        op->attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();

    if (upper_op_name == "pd_op.feed" || upper_op_name == "pd_op.data") {
      skip_dels.insert(op->result(0));
      continue;
    }
    if (upper_op_name == "pd_op.fetch" ||
        upper_op_name == "pd_op.shadow_output") {
      skip_dels.insert(op->operand_source(0));
      continue;
    }
  }
  return skip_dels;
}

// NOTE(zhangbo): For inplace Pass, currently only the kernel_dialect operator
// is supported. Therefore, this function only returns the values in the
// kernel_dialect operator that can be eager deleted.
static std::unordered_map<pir::Operation*, std::unordered_set<pir::Value>>
GetEagerDeletionValues(pir::Block* block) {
  std::unordered_set<pir::Value> skip_dels = GetSkipDeletionValues(block);

  std::unordered_map<pir::Value, pir::Operation*> del_value_2_op;
  for (auto& op : *block) {
    std::string upper_op_name = op->name();
    if (op->dialect()->name().compare(paddle::dialect::KernelDialect::name()) ==
        0) {
      IR_ENFORCE(op->attributes().count("op_name") > 0,
                 "kernel_dialect op should own an 'op_name' attribute.");
      upper_op_name = op->attributes()
                          .at("op_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();
    }

    for (size_t i = 0; i < op->num_operands(); ++i) {
      auto input = op->operand_source(i);
      if (skip_dels.count(input) > 0 || !input || !CanBeDeleted(input) ||
          IsNoNeedBuffer(op, input)) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << upper_op_name << ") can not be deleted.";
        VLOG(8) << " -- skip dels: " << skip_dels.count(input);
        VLOG(8) << " -- value is null: " << !input;
        VLOG(8) << " -- can be deleted: " << !CanBeDeleted(input);
        VLOG(8) << " -- is no_need_buffer: " << IsNoNeedBuffer(op, input);
        continue;
      }
      del_value_2_op[input] = op;
    }

    for (size_t i = 0; i < op->num_results(); ++i) {
      pir::Value output = op->result(i);
      if (output && CanBeDeleted(output)) {
        del_value_2_op[output] = op;
      }
    }
  }

  std::unordered_map<pir::Operation*, std::unordered_set<pir::Value>>
      eager_dels;
  for (auto& kv : del_value_2_op) {
    eager_dels[kv.second].insert(kv.first);
  }

  return eager_dels;
}

static std::unordered_map<pir::Operation*, std::string> GetInplaceOps(
    pir::Block* block) {
  const auto eager_dels = GetEagerDeletionValues(block);

  std::unordered_map<pir::Operation*, std::string> inplace_ops;

  std::unordered_set<pir::Value> visited_values;
  std::unordered_set<pir::Value> reused_input_values;
  std::unordered_set<pir::Value> reused_output_values;

  for (auto& op : *block) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      visited_values.insert(op->operand_source(i));
    }

    if (op->dialect()->name().compare(paddle::dialect::KernelDialect::name()) !=
        0) {
      VLOG(6) << op->name()
              << "is not a kernel_dialect op, inplace only support "
                 "kernel_dialect operators";
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    auto upper_op_attrs = op->attributes();
    auto upper_op_name =
        upper_op_attrs.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
    VLOG(6) << "analyse op: " << upper_op_name;

    // NOTE(zhangbo): add_grad cpu kernel can't do inplace, for the reason shown
    // in the function: CommonElementwiseBroadcastBackward
    // (paddle/phi/kernels/funcs/elementwise_grad_base.h)
    if ((upper_op_name == "pd_op.add_grad") &&
        (upper_op_attrs.at("kernel_key")
             .dyn_cast<paddle::dialect::KernelAttribute>()
             .data()
             .backend() == phi::Backend::CPU)) {
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    if (upper_op_attrs.count("is_inplace") != 0 &&
        upper_op_attrs.at("is_inplace").dyn_cast<pir::BoolAttribute>().data()) {
      VLOG(6) << upper_op_name << " is already an inplace op.";
      for (size_t i = 0; i < op->num_operands(); ++i) {
        reused_input_values.insert(op->operand_source(i));
      }
      for (size_t i = 0; i < op->num_results(); ++i) {
        reused_output_values.insert(op->result(i));
        visited_values.insert(op->result(i));
      }
      continue;
    }

    pir::OpInfo upper_inplace_op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(upper_op_name + "_");

    if (eager_dels.count(op) == 0 || (!upper_inplace_op_info)) {
      VLOG(6) << upper_op_name
              << "'s value can't delete or doesn't have inplace op, so that "
                 "can't do inplace.";
      for (size_t i = 0; i < op->num_results(); ++i) {
        visited_values.insert(op->result(i));
      }
      continue;
    }

    auto upper_inplace_op_interface =
        upper_inplace_op_info
            .GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
    PADDLE_ENFORCE_NOT_NULL(
        upper_inplace_op_interface,
        phi::errors::PreconditionNotMet(
            "can not find OpYamlInfoInterface from [%s]", upper_op_name + "_"));
    paddle::dialect::OpYamlInfoParser upper_inplace_op_info_parser(
        upper_inplace_op_interface->get_op_info_());
    std::unordered_map<uint32_t, uint32_t> inplace_out_2_in =
        upper_inplace_op_info_parser.GetInplaceIdMap();

    bool can_do_inplace = true;
    for (auto& kv : inplace_out_2_in) {
      uint32_t out_slot = kv.first;
      uint32_t in_slot = kv.second;
      if ((in_slot >= op->num_operands()) || (out_slot >= op->num_results()) ||
          (!CanDoInplace(eager_dels.at(op),
                         op->operand_source(in_slot),
                         op->result(out_slot))) ||
          (visited_values.count(op->result(out_slot)) > 0) ||
          (!CanBeDeleted(op->result(out_slot))) ||
          (reused_input_values.count(op->operand_source(in_slot)) > 0) ||
          (reused_output_values.count(op->result(out_slot)) > 0)) {
        can_do_inplace = false;
        VLOG(6) << upper_op_name
                << "'s value has been visited or reused by other inplace op, "
                   "so that can't do inplace.";
        VLOG(8) << " -- operand " << in_slot << " and result " << out_slot
                << " can do inplace: "
                << CanDoInplace(eager_dels.at(op),
                                op->operand_source(in_slot),
                                op->result(out_slot));
        VLOG(8) << " -- result " << out_slot << " visited: "
                << (visited_values.count(op->result(out_slot)) > 0);
        VLOG(8) << " -- operand " << in_slot << " has been reused: "
                << (reused_input_values.count(op->operand_source(in_slot)) > 0);
        VLOG(8) << " -- result " << out_slot << " has been reused: "
                << (reused_output_values.count(op->result(out_slot)) > 0);
        break;
      }
    }
    if (can_do_inplace) {
      inplace_ops[op] = upper_op_name + "_";
      for (auto& kv : inplace_out_2_in) {
        reused_input_values.insert(op->operand_source(kv.second));
        reused_output_values.insert(op->result(kv.first));
      }
      VLOG(6) << upper_op_name
              << " will change to inplace version op: " << upper_op_name + "_";
    }

    for (size_t i = 0; i < op->num_results(); ++i) {
      visited_values.insert(op->result(i));
    }
  }
  return inplace_ops;
}
}  // namespace details

class InplacePass : public pir::Pass {
 public:
  InplacePass() : pir::Pass("InplacePass", 3) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "InplacePass should run on module op.");
    auto* block = module_op.block();

    auto inplace_ops = details::GetInplaceOps(block);

    for (auto kv : inplace_ops) {
      VLOG(6) << "Do inplace for: "
              << kv.first->attributes()
                     .at("op_name")
                     .dyn_cast<pir::StrAttribute>()
                     .AsString();
      pir::Block::iterator insert_pos =
          std::find(block->begin(), block->end(), kv.first);
      IR_ENFORCE(insert_pos != block->end(),
                 "Operator %s not found in block.",
                 kv.first->name());

      kv.first->set_attribute(
          "op_name",
          pir::StrAttribute::get(pir::IrContext::Instance(), kv.second));
      kv.first->set_attribute(
          "is_inplace",
          pir::BoolAttribute::get(pir::IrContext::Instance(), true));
    }
    LOG_FIRST_N(INFO, 1)
        << "Apply inplace pass on lowering ::ir::Program to Kernel Dialect.";
  }

  bool CanApplyOn(pir::Operation* op) const override {
    return op->isa<::pir::ModuleOp>() && op->num_regions() > 0;
  }
};

namespace pir {

std::unique_ptr<pir::Pass> CreateInplacePass() {
  return std::make_unique<InplacePass>();
}

}  // namespace pir

REGISTER_IR_PASS(inplace, InplacePass);
