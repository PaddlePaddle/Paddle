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

// #include <sstream>
// #include <string>

#include "paddle/fluid/primitive/base/decomp_trans.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/program.h"

PHI_DECLARE_bool(prim_skip_dynamic);

using paddle::dialect::DenseTensorType;
using paddle::dialect::SelectedRowsType;

namespace paddle {

using Program = pir::Program;

static bool find_value(const std::vector<int64_t>& vec, int64_t value) {
  if (std::find(vec.begin(), vec.end(), value) != vec.end()) {
    return true;
  } else {
    return false;
  }
}

static const phi::DDim& GetValueDims(pir::Value value) {
  if (value.type().isa<DenseTensorType>()) {
    return value.type().dyn_cast<DenseTensorType>().dims();
  } else if (value.type().isa<SelectedRowsType>()) {
    return value.type().dyn_cast<SelectedRowsType>().dims();
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get shape for dense "
        "tensor."));
  }
}

static bool check_dynamic_shape(const pir::OpOperand& item,
                                const pir::Operation& op) {
  auto dims = GetValueDims(item.source());
  std::vector<int64_t> shape = common::vectorize<int64_t>(dims);
  if (find_value(shape, -1)) {
    LOG(WARNING)
        << "[Prim] Decomp op does not support dynamic shape -1, but got "
           "shape "
        << dims << "in inputs of op " << op.name();
    return true;
  } else {
    return false;
  }
}

bool has_decomp_rule(const pir::Operation& op) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(op.name());
  auto decomp_interface_impl =
      op_info.GetInterfaceImpl<paddle::dialect::DecompInterface>();
  if (decomp_interface_impl == nullptr) return false;
  return true;
}

bool DecompProgram::check_decomp_dynamic_shape(pir::Operation* op) {
  for (auto item : op->operands()) {
    auto value = item.source();
    // check if initialized in case of optional input.
    if (value.impl() && value.type().storage()) {
      pir::Operation* prev_op = value.dyn_cast<pir::OpResult>().owner();
      if (prev_op->name() == "builtin.combine") {
        for (pir::OpOperand& sub_item : prev_op->operands()) {
          if (check_dynamic_shape(sub_item, *op)) {
            return true;
          }
        }
      } else {
        if (check_dynamic_shape(item, *op)) {
          return true;
        }
      }
      // PADDLE_ENFORCE_NOT_NULL(
      //     prev_op, platform::errors::PreconditionNotMet("prev_op should not
      //     be null"));
    }
  }
  return false;
}

void DecompProgram::check_decomp_outputs(
    const std::string& op_name,
    const std::vector<pir::OpResult>& orig_outs,
    const std::vector<pir::OpResult>& decomp_outs) {
  return;
}

std::vector<pir::OpResult> DecompProgram::format_decomp_res(
    const std::string& op_name,
    const std::vector<pir::OpResult>& orig_outs,
    const std::vector<std::vector<pir::OpResult>>& decomp_outs) {
  PADDLE_ENFORCE_EQ(orig_outs.size(),
                    decomp_outs.size(),
                    paddle::platform::errors::PreconditionNotMet(
                        "For op %s, its origin output num %d is not equal to "
                        "decomp output num %d ",
                        op_name,
                        orig_outs.size(),
                        decomp_outs.size()));
  std::vector<pir::OpResult> new_decomp_outs(orig_outs.size());
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (orig_outs[i]) {
      PADDLE_ENFORCE_EQ(decomp_outs[i].size(),
                        1,
                        paddle::platform::errors::PreconditionNotMet(
                            "For op %s, each element of decomp output num must "
                            "be 1, but num of index %d is %d ",
                            op_name,
                            i,
                            decomp_outs[i].size()));
      new_decomp_outs[i] = decomp_outs[i][0];
    }
  }
  return new_decomp_outs;
}

std::vector<pir::OpResult> DecompProgram::construct_dst_vars(
    const std::string& op_name,
    const std::vector<pir::OpResult>& orig_outs,
    const std::vector<pir::OpResult>& decomp_outs,
    std::unordered_map<pir::OpResult, int> orig_vars_dict) {
  std::vector<pir::OpResult> tar_vars(src_vars_.size());
  PADDLE_ENFORCE_EQ(orig_outs.size(),
                    decomp_outs.size(),
                    paddle::platform::errors::PreconditionNotMet(
                        "For op %s, its origin output num %d is not equal to "
                        "decomp output num %d ",
                        op_name,
                        orig_outs.size(),
                        decomp_outs.size()));
  for (size_t i = 0; i < orig_outs.size(); i++) {
    VLOG(4) << "decomp construct idx -------- " << i;
    if (orig_vars_dict.find(orig_outs[i]) != orig_vars_dict.end()) {
      VLOG(4) << "decomp construct in idx -------- " << i;
      tar_vars[orig_vars_dict[orig_outs[i]]] = decomp_outs[i];
    }
  }
  return tar_vars;
}

bool DecompProgram::enable_decomp_by_filter(const std::string& op_name) {
  bool flag = true;

  if (whitelist_.size() > 0) {
    if (whitelist_.find(op_name) == whitelist_.end()) {
      flag = false;
    }
  }
  if (blacklist_.size() > 0) {
    if (blacklist_.find(op_name) != blacklist_.end()) {
      flag = false;
    }
  }
  return flag;
}

std::vector<std::vector<pir::OpResult>> call_decomp_rule(pir::Operation* op) {
  paddle::dialect::DecompInterface decomp_interface =
      op->dyn_cast<paddle::dialect::DecompInterface>();
  PADDLE_ENFORCE(
      decomp_interface,
      phi::errors::InvalidArgument(
          "The decomp function is not registered in %s op ", op->name()));
  std::vector<std::vector<pir::OpResult>> decomp_res =
      decomp_interface.Decomp(op);
  return decomp_res;
}

DecompProgram::DecompProgram(pir::Program* program,
                             const std::vector<pir::OpResult>& src_vars,
                             const std::set<std::string>& blacklist,
                             const std::set<std::string>& whitelist)
    : program_(program),
      src_vars_(src_vars),
      blacklist_(blacklist),
      whitelist_(whitelist) {}

std::vector<pir::OpResult> DecompProgram::decomp_program() {
  std::ostringstream print_stream;
  std::unordered_map<pir::OpResult, int> orig_vars_dict;
  for (size_t i = 0; i < src_vars_.size(); i++) {
    orig_vars_dict[src_vars_[i]] = static_cast<int>(i);
  }
  program_->Print(print_stream);
  VLOG(4) << "program in sink decomp ------" << print_stream.str();
  if (!paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
    return src_vars_;
  }
  std::vector<pir::OpResult> tar_vars(src_vars_.size());
  pir::Block* block = program_->block();
  std::vector<pir::Operation*> ops_list;
  for (auto& op : *block) {
    ops_list.push_back(&op);
  }
  for (size_t i = 0; i < ops_list.size(); i++) {
    auto op = ops_list[i];
    bool enable_prim =
        has_decomp_rule(*op) && enable_decomp_by_filter(op->name());
    if (enable_prim && FLAGS_prim_skip_dynamic &&
        check_decomp_dynamic_shape(op)) {
      enable_prim = false;
    }
    VLOG(4) << "enable_prim flag ======= " << enable_prim;
    if (enable_prim) {
      VLOG(4) << "decomp op name ======= " << op->name();

      auto& builder = *(paddle::dialect::ApiBuilder::Instance().GetBuilder());
      builder.set_insertion_point(op);
      std::vector<std::vector<pir::OpResult>> decomp_res = call_decomp_rule(op);
      std::vector<pir::OpResult> orig_outs = op->results();
      std::vector<pir::OpResult> standard_decomp_res =
          format_decomp_res(op->name(), orig_outs, decomp_res);
      tar_vars = construct_dst_vars(
          op->name(), orig_outs, standard_decomp_res, orig_vars_dict);

      VLOG(4) << "decomp out size ======= " << decomp_res.size();
      op->ReplaceAllUsesWith(standard_decomp_res);
      std::ostringstream print_stream2;
      program_->Print(print_stream2);
      VLOG(4) << "program out sink decomp ------ index " << i << ". "
              << print_stream2.str();
      bool remove_op = true;
      for (auto& item : op->results()) {
        if (item.HasOneUse()) {
          remove_op = false;
          break;
        }
      }
      VLOG(4) << "program remove op ----------- " << remove_op << ". "
              << op->name();
      if (remove_op) {
        auto op_iter = std::find(block->begin(), block->end(), *op);
        block->erase(op_iter);
      }
    }
  }
  for (size_t i = 0; i < tar_vars.size(); i++) {
    if (!tar_vars[i]) {
      VLOG(4) << "assign tar_vars ===========  " << i;
      tar_vars[i] = src_vars_[i];
    }
  }
  auto& builder = *(paddle::dialect::ApiBuilder::Instance().GetBuilder());
  builder.SetInsertionPointToEnd(block);
  std::ostringstream print_stream3;
  program_->Print(print_stream3);
  VLOG(4) << "program out final ************ " << print_stream3.str();
  return tar_vars;
}

}  // namespace paddle
