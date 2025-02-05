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

#include "paddle/common/flags.h"
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
#include "paddle/fluid/pir/transforms/general/inplace_pass.h"
#include "paddle/fluid/pir/utils/general_functions.h"
#include "paddle/pir/include/core/builtin_op.h"
#include "paddle/pir/include/core/op_operand.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_registry.h"

COMMON_DECLARE_string(ir_inplace_kernel_blacklist);

namespace {

using TensorType = paddle::dialect::AllocatedDenseTensorType;

std::unordered_set<std::string> IgnoreShapeCheckOps = {
    paddle::dialect::ReshapeOp::name(),
    paddle::dialect::SqueezeOp::name(),
    paddle::dialect::UnsqueezeOp::name(),
};

std::unordered_set<std::string> RelaxShapeCheckOps = {
    paddle::dialect::ReshapeGradOp::name(),
    paddle::dialect::AddGradOp::name(),
};

// NOTE(zhangbo): Which kind of value can be deleted?
// (1) Value's type needs to be AllocatedDenseTensorType or
// AllocatedSelectedRowsType; (2) Value's is not persistable.
bool CanBeDeleted(pir::Value value) {
  if (!value.type()) {
    return false;
  }
  if (!value.type().isa<TensorType>() &&
      !value.type().isa<paddle::dialect::AllocatedSelectedRowsType>()) {
    return false;
  }
  auto persist_attr = value.attribute<pir::BoolAttribute>(kAttrIsPersistable);
  return !(persist_attr && persist_attr.data());
}

bool IsLastUser(const pir::Value& value,
                const std::unordered_map<pir::Value, size_t>& use_count_map,
                const std::unordered_map<pir::Value, pir::Value>& inplace_map) {
  auto current_value = value;
  while (use_count_map.at(current_value) == 0) {
    if (inplace_map.count(current_value) == 0) {
      return true;
    }
    current_value = inplace_map.at(current_value);
  }
  return false;
}

bool CanDoInplace(const std::unordered_set<pir::Value>& eager_dels,
                  pir::Value input,
                  pir::Value output,
                  const std::string& op_name) {
  if (!input.type() || !output.type() || input.isa<pir::BlockArgument>()) {
    return false;
  }

  if (input.defining_op()->num_regions() > 0) {
    auto cf_op = input.defining_op();
    std::vector<pir::Value> related_values;
    for (size_t i = 0; i < cf_op->num_operands(); i++) {
      related_values.push_back(cf_op->operand_source(i));
    }
    for (size_t i = 0; i < cf_op->num_regions(); i++) {
      auto& region = cf_op->region(i);
      for (auto& block : region)
        if (!block.empty() && block.back().isa<pir::YieldOp>()) {
          auto& yield_op = block.back();
          for (size_t i = 0; i < yield_op.num_operands(); i++) {
            related_values.push_back(yield_op.operand_source(i));
          }
        }
    }

    for (auto& value : related_values) {
      auto persist_attr =
          value.attribute<pir::BoolAttribute>(kAttrIsPersistable);
      if (persist_attr && persist_attr.data()) {
        VLOG(9) << "     -- input tensor is shared with a persistable tensor, "
                   "can't do inplace";
        return false;
      }
    }
  }

  if (input.type().isa<TensorType>() && output.type().isa<TensorType>()) {
    auto input_alloc_tensor_type = input.type().dyn_cast<TensorType>();
    auto output_alloc_tensor_type = output.type().dyn_cast<TensorType>();

    if (input_alloc_tensor_type.dtype() != output_alloc_tensor_type.dtype()) {
      VLOG(9) << "     -- input's dtype != output's dtype, can't do inplace";
      return false;
    }

    if (IgnoreShapeCheckOps.count(op_name) > 0 &&
        eager_dels.count(input) != 0) {
      VLOG(9) << "     -- reshape, squeeze, unsqueeze do not need check shape, "
                 "can do inplace";
      return true;
    }

    auto is_numel_equal = [](const TensorType& in,
                             const TensorType& out) -> bool {
      int64_t in_numel = 1;
      int64_t out_numel = 1;
      for (int i = 0; i < in.dims().size(); i++) {
        if (in.dims()[i] == -1 && in.dims().size() == 1) {
          VLOG(9) << "     -- input's shape has -1 and dim size is 1, can't "
                     "do inplace";
          return false;
        }
        if (in.dims()[i] == -1 && i != 0) {
          VLOG(9) << "     -- input's shape has -1 and not in first dim, can't "
                     "do inplace";
          return false;
        }
        in_numel *= in.dims()[i];
      }

      for (int i = 0; i < out.dims().size(); i++) {
        if (out.dims()[i] == -1 && out.dims().size() == 1) {
          VLOG(9) << "     -- output's shape has -1 and dim size is 1, can't "
                     "do inplace";
          return false;
        }
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
    auto is_numel_equal_loose_version = [](const TensorType& in,
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
    bool relax = (RelaxShapeCheckOps.count(op_name) > 0);
    if (relax) {
      equal = is_numel_equal_loose_version(input_alloc_tensor_type,
                                           output_alloc_tensor_type);
    } else {
      equal = is_numel_equal(input_alloc_tensor_type, output_alloc_tensor_type);
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
    VLOG(9) << "     -- input not in eager_deletion_vars, can't do inplace";
    return false;
  }
  return true;
}

bool IsNoNeedBuffer(pir::Operation* op, pir::Value value) {
  if (op->dialect()->name() != paddle::dialect::KernelDialect::name()) {
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
          info_interface->get_op_info_(op_name),
          paddle::dialect::IsLegacyOp(op_name));
      auto& no_need_buffer_ids = info_parser.NoNeedBufferIds();
      for (auto no_need_buffer_id : no_need_buffer_ids) {
        if (value == op->operand_source(no_need_buffer_id)) {
          return true;
        }
      }
    }
  }
  return false;
}

// NOTE(zhangbo): pd_op.feed's output and pd_op.fetch's input can not be eager
// deleted.
std::unordered_set<pir::Value> GetSkipDeletionValues(
    const pir::Block& block,
    const std::set<std::string>& no_need_buffer_values) {
  std::unordered_set<pir::Value> skip_dels;
  for (auto& op : block) {
    if (op.name() == "builtin.shadow_output" &&
        no_need_buffer_values.count(op.attributes()
                                        .at("output_name")
                                        .dyn_cast<pir::StrAttribute>()
                                        .AsString()) == 0) {
      skip_dels.insert(op.operand_source(0));
      continue;
    }
    if (op.dialect()->name() != paddle::dialect::KernelDialect::name()) {
      continue;
    }
    PADDLE_ENFORCE_GT(
        op.attributes().count("op_name"),
        0UL,
        common::errors::InvalidArgument(
            "kernel_dialect op should own an 'op_name' attribute."));
    auto upper_op_name =
        op.attributes().at("op_name").dyn_cast<pir::StrAttribute>().AsString();

    if (upper_op_name == "pd_op.feed" || upper_op_name == "pd_op.data" ||
        upper_op_name == "pd_op.shadow_feed") {
      skip_dels.insert(op.result(0));
      continue;
    }
    // TODO(chenxi67) add logic for shadow_feed_tensors op
    if (upper_op_name == "pd_op.fetch") {
      skip_dels.insert(op.operand_source(0));
      continue;
    }
  }
  return skip_dels;
}

// NOTE(zhangbo): For inplace Pass, currently only the kernel_dialect operator
// is supported. Therefore, this function only returns the values in the
// kernel_dialect operator that can be eager deleted.
void GetEagerDelValueOfOp(
    const pir::Block& block,
    const std::unordered_set<pir::Value>& skip_dels,
    std::unordered_map<pir::Value, pir::Operation*>* del_value_2_op) {
  for (auto& op : block) {
    std::string upper_op_name = op.name();
    if (op.dialect()->name() == paddle::dialect::KernelDialect::name()) {
      PADDLE_ENFORCE_GT(
          op.attributes().count("op_name"),
          0UL,
          common::errors::InvalidArgument(
              "kernel_dialect op should own an 'op_name' attribute."));
      upper_op_name = op.attributes()
                          .at("op_name")
                          .dyn_cast<pir::StrAttribute>()
                          .AsString();
    }

    if (upper_op_name == "builtin.shadow_output") {
      continue;
    }

    for (size_t i = 0; i < op.num_operands(); ++i) {
      auto input = op.operand_source(i);
      if (skip_dels.count(input) > 0 || !input || !CanBeDeleted(input) ||
          IsNoNeedBuffer(&op, input)) {
        VLOG(6) << "The " << i << "-th input value of the Operation("
                << upper_op_name << ") can not be deleted.";
        VLOG(8) << " -- skip dels: " << skip_dels.count(input);
        VLOG(8) << " -- value is null: " << !input;
        VLOG(8) << " -- can be deleted: " << !CanBeDeleted(input);
        VLOG(8) << " -- is no_need_buffer: " << IsNoNeedBuffer(&op, input);
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

    if (op.num_regions() > 0) {
      for (size_t i = 0; i < op.num_regions(); ++i) {
        for (const auto& inner_block : op.region(i)) {
          GetEagerDelValueOfOp(inner_block, skip_dels, del_value_2_op);
        }
      }
      VLOG(8) << "GetEagerDelValueOfOp for sub block";
    }
  }
}

std::unordered_map<pir::Operation*, std::unordered_set<pir::Value>>
GetEagerDeletionValues(const pir::Block& block,
                       const std::set<std::string>& no_need_buffer_values) {
  std::unordered_set<pir::Value> skip_dels =
      GetSkipDeletionValues(block, no_need_buffer_values);
  std::unordered_map<pir::Value, pir::Operation*> del_value_2_op;
  GetEagerDelValueOfOp(block, skip_dels, &del_value_2_op);
  std::unordered_map<pir::Operation*, std::unordered_set<pir::Value>>
      eager_dels;
  for (auto& kv : del_value_2_op) {
    eager_dels[kv.second].insert(kv.first);
  }
  return eager_dels;
}

std::unordered_map<pir::Operation*, std::string> GetInplaceOps(
    const pir::Block& block,
    const std::set<std::string>& no_need_buffer_values) {
  const auto eager_dels = GetEagerDeletionValues(block, no_need_buffer_values);

  auto is_no_need_buffer = [&no_need_buffer_values](pir::Operation* op,
                                                    pir::Value value) {
    if (auto shadow_output_op = op->dyn_cast<pir::ShadowOutputOp>()) {
      if (no_need_buffer_values.count(shadow_output_op.attributes()
                                          .at("output_name")
                                          .dyn_cast<pir::StrAttribute>()
                                          .AsString())) {
        return true;
      }
    }
    return IsNoNeedBuffer(op, value);
  };
  auto use_count_map = [&is_no_need_buffer](const pir::Block& block) {
    std::unordered_map<pir::Value, size_t> use_count_map;
    for (auto& op : block) {
      for (auto value : op.results()) {
        size_t use_count = 0;
        for (auto it = value.use_begin(); it != value.use_end(); ++it) {
          use_count += is_no_need_buffer(it->owner(), value) ? 0 : 1;
        }
        use_count_map[value] = use_count;
      }
    }
    return use_count_map;
  }(block);
  std::unordered_map<pir::Value, pir::Value> inplace_map;

  std::unordered_map<pir::Operation*, std::string> inplace_ops;

  std::unordered_set<pir::Value> visited_values;

  for (auto& op : block) {
    for (size_t i = 0; i < op.num_operands(); ++i) {
      visited_values.insert(op.operand_source(i));
      use_count_map[op.operand_source(i)] -=
          is_no_need_buffer(&op, op.operand_source(i)) ? 0 : 1;
    }

    if (op.dialect()->name() != paddle::dialect::KernelDialect::name()) {
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
      auto op_info =
          pir::IrContext::Instance()->GetRegisteredOpInfo(upper_op_name);
      auto op_yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      paddle::dialect::OpYamlInfoParser op_info_parser(
          op_yaml_interface->get_op_info_(upper_op_name));
      for (auto [out_slot, in_slot] : op_info_parser.GetInplaceIdMap()) {
        auto out_value = op.result(out_slot);
        auto in_value = op.operand_source(in_slot);
        inplace_map[out_value] = in_value;
      }
      for (auto& result : op.results()) {
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
        common::errors::PreconditionNotMet(
            "can not find OpYamlInfoInterface from [%s]", upper_op_name + "_"));
    paddle::dialect::OpYamlInfoParser upper_inplace_op_info_parser(
        upper_inplace_op_interface->get_op_info_(upper_op_name + "_"));
    std::unordered_map<uint32_t, uint32_t> inplace_out_2_in =
        upper_inplace_op_info_parser.GetInplaceIdMap();

    const auto used_external_values = GetUsedExternalValue(block);

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
          !IsLastUser(op.operand_source(in_slot), use_count_map, inplace_map) ||
          (std::find(used_external_values.begin(),
                     used_external_values.end(),
                     op.operand_source(in_slot)) !=
           used_external_values.end()) ||
          (std::find(used_external_values.begin(),
                     used_external_values.end(),
                     op.result(out_slot)) != used_external_values.end())) {
        can_do_inplace = false;
        VLOG(6) << upper_op_name
                << "'s value has been visited or reused by other inplace op, "
                   "so that can't do inplace when setting relax to :"
                << (RelaxShapeCheckOps.count(upper_op_name) > 0);
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
            << " -- operand " << in_slot << " is last user: "
            << IsLastUser(
                   op.operand_source(in_slot), use_count_map, inplace_map);
        break;
      }
    }
    if (can_do_inplace) {
      inplace_ops[&op] = upper_op_name + "_";
      for (auto& kv : inplace_out_2_in) {
        inplace_map[op.result(kv.first)] = op.operand_source(kv.second);
      }
      VLOG(6) << upper_op_name
              << " will change to inplace version op: " << upper_op_name + "_";
    }

    for (auto& result : op.results()) {
      visited_values.insert(result);
    }
  }
  if (!FLAGS_ir_inplace_kernel_blacklist.empty()) {
    for (auto const& i : inplace_ops) {
      std::cout << i.second << std::endl;
    }
  }
  return inplace_ops;
}
}  // namespace

class InplacePass : public pir::Pass {
 public:
  InplacePass() : pir::Pass("inplace_pass", 3) {}

  explicit InplacePass(const std::set<std::string>& no_need_buffer_values)
      : pir::Pass("inplace_pass", 3) {
    no_need_buffer_values_ = no_need_buffer_values;
  }

  void Run(pir::Operation* op) override {
    int64_t num_rewrites_{0};
    for (size_t i = 0; i < op->num_regions(); ++i) {
      auto& region = op->region(i);
      for (auto& block : region) {
        auto inplace_ops = GetInplaceOps(block, no_need_buffer_values_);

        for (const auto& kv : inplace_ops) {
          VLOG(6) << "Do inplace for: "
                  << kv.first->attributes()
                         .at("op_name")
                         .dyn_cast<pir::StrAttribute>()
                         .AsString();
          pir::Block::Iterator insert_pos =
              std::find(block.begin(), block.end(), *kv.first);
          PADDLE_ENFORCE_NE(
              insert_pos,
              block.end(),
              common::errors::InvalidArgument("Operator %s not found in block.",
                                              kv.first->name()));

          kv.first->set_attribute(
              "op_name",
              pir::StrAttribute::get(pir::IrContext::Instance(), kv.second));
          kv.first->set_attribute(
              "is_inplace",
              pir::BoolAttribute::get(pir::IrContext::Instance(), true));
          num_rewrites_++;
        }
      }
    }
    AddStatistics(num_rewrites_);
  }

 private:
  std::set<std::string> no_need_buffer_values_;
};

namespace pir {

std::unique_ptr<pir::Pass> CreateInplacePass(
    const std::set<std::string>& no_need_buffer_values) {
  return std::make_unique<InplacePass>(no_need_buffer_values);
}

}  // namespace pir

REGISTER_IR_PASS(inplace_pass, InplacePass);
