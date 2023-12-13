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

#include <numeric>
#include <regex>
#include <string>
#include <unordered_set>

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/inplace_pass.h"
#include "paddle/phi/core/flags.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_registry.h"

PHI_DECLARE_string(ir_inplace_kernel_blacklist);

namespace details {

using TensorType = paddle::dialect::AllocatedDenseTensorType;

static std::unordered_set<std::string> ignore_shape_check_ops = {
    paddle::dialect::ReshapeOp::name(),
    paddle::dialect::SqueezeOp::name(),
    paddle::dialect::UnsqueezeOp::name(),
};

static std::unordered_set<std::string> relax_shape_check_ops = {
    paddle::dialect::ReshapeGradOp::name(),
    paddle::dialect::AddGradOp::name(),
};

// NOTE(zhangbo): Which kind of value can be deleted?
// (1) Value's type needs to be AllocatedDenseTensorType or
// AllocatedSelectedRowsType; (2) Value's is not persisable.
static bool CanBeDeleted(pir::Value value) {
  if (!value.type()) {
    return false;
  }
  if (!value.type().isa<TensorType>() &&
      !value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    return false;
  }
  auto persist_attr = value.attribute<pir::BoolAttribute>(kAttrIsPersisable);
  return !(persist_attr && persist_attr.data());
}

static bool CanDoInplace(const std::unordered_set<pir::Value>& eager_dels,
                         pir::Value input,
                         pir::Value output,
                         const std::string& op_name) {
  if (!input.type() || !output.type()) {
    return false;
  }

  if (input.type().isa<TensorType>() && output.type().isa<TensorType>()) {
    auto input_alloc_tensor_type = input.type().dyn_cast<TensorType>();
    auto output_alloc_tensor_type = output.type().dyn_cast<TensorType>();

    if (input_alloc_tensor_type.dtype() != output_alloc_tensor_type.dtype()) {
      VLOG(9) << "     -- input's dtype != output's dtype, can't do inplace";
      return false;
    }

    if (details::ignore_shape_check_ops.count(op_name) > 0 &&
        eager_dels.count(input) != 0) {
      VLOG(9) << "     -- reshape, squeeze, unsqueeze do not need check shape, "
                 "can do inplace";
      return true;
    }

    auto is_numel_euqal = [](const TensorType& in,
                             const TensorType& out) -> bool {
      int64_t in_numel = 1;
      int64_t out_numel = 1;
      for (int i = 0; i < in.dims().size(); i++) {
        if (in.dims()[i] == -1 && i != 0) {
          VLOG(9) << "     -- input's shape has -1 and not in first dim, can't "
                     "do inplace";
          return false;
        }
        in_numel *= in.dims()[i];
      }

      for (int i = 0; i < out.dims().size(); i++) {
        if (out.dims()[i] == -1 && i != 0) {
          VLOG(9)
              << "     -- output's shape has -1 and not in first dim, can't "
                 "do inplace";
          return false;
        }
        out_numel *= out.dims()[i];
      }
      return in_numel == out_numel;
    };
    // In this version, we don't consider the -1 in ddim, we just calculate the
    // result.
    auto is_numel_euqal_loose_version = [](const TensorType& in,
                                           const TensorType& out) -> bool {
      auto calculate_numel = [](const phi::DDim& ddim) -> int64_t {
        int64_t numel = 1;
        for (int i = 0; i < ddim.size(); i++) {
          numel *= ddim[i];
        }
        return numel;
      };
      int64_t in_numel = calculate_numel((in.dims()));
      int64_t out_numel = calculate_numel((out.dims()));
      VLOG(10) << "in: " << in_numel << ", out: " << out_numel;
      return in_numel == out_numel;
    };
    bool equal = false;
    bool relax = (details::relax_shape_check_ops.count(op_name) > 0);
    if (relax) {
      equal = is_numel_euqal_loose_version(input_alloc_tensor_type,
                                           output_alloc_tensor_type);
    } else {
      equal = is_numel_euqal(input_alloc_tensor_type, output_alloc_tensor_type);
    }

    if (!equal) {
      VLOG(9) << "     -- input's numel != output's numel, can't do inplace";
      return false;
    }
  } else if (input.type() != output.type()) {
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
          info_interface->get_op_info_(), paddle::dialect::IsLegacyOp(op_name));
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
    if (op.dialect()->name().compare(paddle::dialect::KernelDialect::name()) !=
        0) {
      continue;
    }
    IR_ENFORCE(op.attributes().count("op_name") > 0,
               "kernel_dialect op should own an 'op_name' attribute.");
    auto upper_op_name =
        op.attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();

    if (upper_op_name == "pd_op.feed" || upper_op_name == "pd_op.data" ||
        upper_op_name == "pd_op.shadow_feed") {
      skip_dels.insert(op.result(0));
      continue;
    }
    if (upper_op_name == "pd_op.fetch" ||
        upper_op_name == "builtin.shadow_output") {
      skip_dels.insert(op.operand_source(0));
      continue;
    }
  }
  return skip_dels;
}

// NOTE(zhangbo): For inplace Pass, currently only the kernel_dialect operator
// is supported. Therefore, this function only returns the values in the
// kernel_dialect operator that can be eager deleted.
static void GetEagerDelValueOfOp(
    pir::Block* block,
    const std::unordered_set<pir::Value>& skip_dels,
    std::unordered_map<pir::Value, pir::Operation*>* del_value_2_op) {
  for (auto& op : *block) {
    std::string upper_op_name = op.name();
    if (op.dialect()->name().compare(paddle::dialect::KernelDialect::name()) ==
        0) {
      IR_ENFORCE(op.attributes().count("op_name") > 0,
                 "kernel_dialect op should own an 'op_name' attribute.");
      upper_op_name = op.attributes()
                          .at("op_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();
    }

    for (size_t i = 0; i < op.num_operands(); ++i) {
      auto input = op.operand_source(i);
      if (skip_dels.count(input) > 0 || !input || !CanBeDeleted(input)) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << upper_op_name << ") can not be deleted.";
        VLOG(8) << " -- skip dels: " << skip_dels.count(input);
        VLOG(8) << " -- value is null: " << !input;
        VLOG(8) << " -- can be deleted: " << !CanBeDeleted(input);
        continue;
      }
      (*del_value_2_op)[input] = &op;
    }

    for (auto& result : op.results()) {
      pir::Value output = result;
      if (output && CanBeDeleted(output)) {
        (*del_value_2_op)[output] = &op;
      }
    }

    if (op.isa<paddle::dialect::IfOp>()) {
      auto if_op = op.dyn_cast<paddle::dialect::IfOp>();
      GetEagerDelValueOfOp(&if_op.true_block(), skip_dels, del_value_2_op);
      VLOG(8) << "GetEagerDelValueOfOp for IfOp true block";
      GetEagerDelValueOfOp(&if_op.false_block(), skip_dels, del_value_2_op);
      VLOG(8) << "GetEagerDelValueOfOp for IfOp false block";
    }
  }
}

static std::unordered_map<pir::Operation*, std::unordered_set<pir::Value>>
GetEagerDeletionValues(pir::Block* block) {
  std::unordered_set<pir::Value> skip_dels = GetSkipDeletionValues(block);

  std::unordered_map<pir::Value, pir::Operation*> del_value_2_op;
  GetEagerDelValueOfOp(block, skip_dels, &del_value_2_op);

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
    for (size_t i = 0; i < op.num_operands(); ++i) {
      visited_values.insert(op.operand_source(i));
    }

    if (op.dialect()->name().compare(paddle::dialect::KernelDialect::name()) !=
        0) {
      VLOG(6) << op.name()
              << "is not a kernel_dialect op, inplace only support "
                 "kernel_dialect operators";
      for (auto& result : op.results()) {
        visited_values.insert(result);
      }
      continue;
    }

    auto upper_op_attrs = op.attributes();
    auto upper_op_name =
        upper_op_attrs.at("op_name").dyn_cast<pir::StrAttribute>().AsString();
    VLOG(6) << "analyse op: " << upper_op_name;

    // NOTE(zhangbo): add_grad cpu kernel can't do inplace, for the reason shown
    // in the function: CommonElementwiseBroadcastBackward
    // (paddle/phi/kernels/funcs/elementwise_grad_base.h)
    if ((upper_op_name == "pd_op.add_grad" ||
         upper_op_name == "pd_op.subtract_grad") &&
        (upper_op_attrs.at("kernel_key")
             .dyn_cast<paddle::dialect::KernelAttribute>()
             .data()
             .backend() == phi::Backend::CPU)) {
      for (auto& result : op.results()) {
        visited_values.insert(result);
      }
      continue;
    }

    if (upper_op_attrs.count("is_inplace") != 0 &&
        upper_op_attrs.at("is_inplace").dyn_cast<pir::BoolAttribute>().data()) {
      VLOG(6) << upper_op_name << " is already an inplace op.";
      for (size_t i = 0; i < op.num_operands(); ++i) {
        reused_input_values.insert(op.operand_source(i));
      }
      for (auto& result : op.results()) {
        reused_output_values.insert(result);
        visited_values.insert(result);
      }
      continue;
    }

    pir::OpInfo upper_inplace_op_info =
        pir::IrContext::Instance()->GetRegisteredOpInfo(upper_op_name + "_");

    std::regex reg(",");
    std::unordered_set<std::string> elems{
        std::sregex_token_iterator(FLAGS_ir_inplace_kernel_blacklist.begin(),
                                   FLAGS_ir_inplace_kernel_blacklist.end(),
                                   reg,
                                   -1),
        std::sregex_token_iterator()};
    elems.erase("");

    if (elems.count(upper_op_name)) {
      VLOG(6) << upper_op_name
              << "'s value can't delete or doesn't have inplace op, so that "
                 "can't do inplace.";
      for (size_t i = 0; i < op.num_results(); ++i) {
        visited_values.insert(op.result(i));
      }
      continue;
    }
    if (eager_dels.count(&op) == 0 || (!upper_inplace_op_info) ||
        upper_op_name == "pd_op.transpose") {
      // NOTE(wanghuancoder): pd_op.transpose is not an
      // inplace op, only strided transpose support
      // inplace in dygraph
      VLOG(6) << upper_op_name
              << "'s value can't delete or doesn't have inplace op, so that "
                 "can't do inplace.";
      for (auto& result : op.results()) {
        visited_values.insert(result);
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
      if ((in_slot >= op.num_operands()) || (out_slot >= op.num_results()) ||
          (!CanDoInplace(eager_dels.at(&op),
                         op.operand_source(in_slot),
                         op.result(out_slot),
                         upper_op_name)) ||
          (visited_values.count(op.result(out_slot)) > 0) ||
          (!CanBeDeleted(op.result(out_slot))) ||
          (reused_input_values.count(op.operand_source(in_slot)) > 0) ||
          (reused_output_values.count(op.result(out_slot)) > 0)) {
        can_do_inplace = false;
        VLOG(6) << upper_op_name
                << "'s value has been visited or reused by other inplace op, "
                   "so that can't do inplace when setting relax to :"
                << (details::relax_shape_check_ops.count(upper_op_name) > 0);
        VLOG_IF(
            8, ((in_slot < op.num_operands()) && (out_slot < op.num_results())))
            << " -- operand " << in_slot << " and result " << out_slot
            << " can do inplace: "
            << CanDoInplace(eager_dels.at(&op),
                            op.operand_source(in_slot),
                            op.result(out_slot),
                            upper_op_name);
        VLOG_IF(8, out_slot < op.num_results())
            << " -- result " << out_slot
            << " visited: " << (visited_values.count(op.result(out_slot)) > 0);
        VLOG_IF(8, in_slot < op.num_operands())
            << " -- operand " << in_slot << " has been reused: "
            << (reused_input_values.count(op.operand_source(in_slot)) > 0);
        VLOG_IF(8, out_slot < op.num_results())
            << " -- result " << out_slot << " has been reused: "
            << (reused_output_values.count(op.result(out_slot)) > 0);
        break;
      }
    }
    if (can_do_inplace) {
      inplace_ops[&op] = upper_op_name + "_";
      for (auto& kv : inplace_out_2_in) {
        reused_input_values.insert(op.operand_source(kv.second));
        reused_output_values.insert(op.result(kv.first));
      }
      VLOG(6) << upper_op_name
              << " will change to inplace version op: " << upper_op_name + "_";
    }

    for (auto& result : op.results()) {
      visited_values.insert(result);
    }
  }
  if (!FLAGS_ir_inplace_kernel_blacklist.empty()) {
    for (auto i : inplace_ops) {
      std::cout << i.second << std::endl;
    }
  }
  return inplace_ops;
}
}  // namespace details

class InplacePass : public pir::Pass {
 public:
  InplacePass() : pir::Pass("inplace_pass", 3) {}

  void Run(pir::Operation* op) override {
    auto module_op = op->dyn_cast<pir::ModuleOp>();
    IR_ENFORCE(module_op, "inplace_pass should run on module op.");
    auto& block = module_op.block();

    auto inplace_ops = details::GetInplaceOps(&block);
    int64_t num_rewrites_{0};
    for (auto kv : inplace_ops) {
      VLOG(6) << "Do inplace for: "
              << kv.first->attributes()
                     .at("op_name")
                     .dyn_cast<pir::StrAttribute>()
                     .AsString();
      pir::Block::Iterator insert_pos =
          std::find(block.begin(), block.end(), *kv.first);
      IR_ENFORCE(insert_pos != block.end(),
                 "Operator %s not found in block.",
                 kv.first->name());

      kv.first->set_attribute(
          "op_name",
          pir::StrAttribute::get(pir::IrContext::Instance(), kv.second));
      kv.first->set_attribute(
          "is_inplace",
          pir::BoolAttribute::get(pir::IrContext::Instance(), true));
      num_rewrites_++;
    }
    PrintStatistics(num_rewrites_);
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

REGISTER_IR_PASS(inplace_pass, InplacePass);
