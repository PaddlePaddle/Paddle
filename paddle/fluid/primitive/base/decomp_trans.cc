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

#include "paddle/fluid/primitive/base/decomp_trans.h"
#include "paddle/fluid/pir/dialect/operator/ir/api_builder.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/prim/utils/utils.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/program.h"

PHI_DECLARE_bool(prim_skip_dynamic);

using paddle::dialect::DenseTensorType;
using paddle::dialect::SelectedRowsType;

namespace paddle {

using Program = pir::Program;

// some outputs like xshape will no longer used after decomp, and those outputs
// will skip checking.
std::unordered_set<std::string> decomp_op_contain_none = {"pd_op.squeeze",
                                                          "pd_op.unsqueeze",
                                                          "pd_op.flatten",
                                                          "pd_op.batch_norm",
                                                          "pd_op.batch_norm_"};

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
        "[Prim] Currently, we can only get shape for dense "
        "tensor."));
  }
}

static phi::DataType GetValueDtype(pir::Value value) {
  if (value.type().isa<DenseTensorType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<DenseTensorType>().dtype());
  } else if (value.type().isa<SelectedRowsType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<SelectedRowsType>().dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get phi::DataType from DenseTensorType and "
        "SelectedRowsType."));
  }
}

static bool check_dynamic_shape(const pir::OpOperand& item,
                                const pir::Operation& op) {
  auto dims = GetValueDims(item.source());
  std::vector<int64_t> shape = common::vectorize<int64_t>(dims);
  if (find_value(shape, -1)) {
    LOG(WARNING)
        << "[Prim] Decomp op does not support dynamic shape -1, but got "
           "shape ["
        << dims << "] in inputs of op " << op.name();
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
    if (!paddle::dialect::IsEmptyValue(value)) {
      pir::Operation* prev_op = value.defining_op();
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
    }
  }
  return false;
}

void DecompProgram::check_decomp_outputs(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<pir::Value>& decomp_outs) {
  bool skip_invalid_op_check =
      decomp_op_contain_none.find(op_name) != decomp_op_contain_none.end();
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (skip_invalid_op_check &&
        paddle::dialect::IsEmptyValue(decomp_outs[i])) {
      VLOG(4) << "[Prim] Decomp op skip check of " << i
              << "-index output of op " << op_name;
    } else {
      PADDLE_ENFORCE(
          !paddle::dialect::IsEmptyValue(orig_outs[i]),
          paddle::platform::errors::PreconditionNotMet(
              "[Prim] For op %s, its origin %d-index output is invalid",
              op_name,
              i));
      PADDLE_ENFORCE(
          !paddle::dialect::IsEmptyValue(decomp_outs[i]),
          paddle::platform::errors::PreconditionNotMet(
              "[Prim] For op %s, its decomp %d-index output is invalid",
              op_name,
              i));
      auto orig_dtype = GetValueDtype(orig_outs[i]);
      auto decomp_dtype = GetValueDtype(decomp_outs[i]);

      PADDLE_ENFORCE(orig_dtype == decomp_dtype,
                     paddle::platform::errors::PreconditionNotMet(
                         "[Prim] For op %s, its origin %d-index output dtype "
                         "%s is not equal to "
                         "decomp output dtype %s ",
                         op_name,
                         i,
                         orig_dtype,
                         decomp_dtype));

      auto orig_dim = GetValueDims(orig_outs[i]);
      auto decomp_dim = GetValueDims(decomp_outs[i]);
      std::vector<int64_t> shape = common::vectorize<int64_t>(orig_dim);
      if (find_value(common::vectorize<int64_t>(orig_dim), -1)) {
        LOG(WARNING)
            << "[Prim] Decomp op does not support dynamic shape -1, but got "
               "shape ["
            << orig_dim << "] in " << i << "-index output of origin op "
            << op_name;
      }
      if (find_value(common::vectorize<int64_t>(decomp_dim), -1)) {
        LOG(WARNING)
            << "[Prim] Decomp op does not support dynamic shape -1, but got "
               "shape ["
            << decomp_dim << "] in " << i << "-index output of decomp op "
            << op_name;
      }

      PADDLE_ENFORCE(orig_dim == decomp_dim,
                     paddle::platform::errors::PreconditionNotMet(
                         "[Prim] For op %s, its origin %d-index output shape "
                         "[%s] is not equal to "
                         "decomp output shape [%s] ",
                         op_name,
                         i,
                         orig_dim,
                         decomp_dim));
    }
  }
  return;
}

std::vector<pir::Value> DecompProgram::format_decomp_res(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<std::vector<pir::Value>>& decomp_outs) {
  PADDLE_ENFORCE_EQ(
      orig_outs.size(),
      decomp_outs.size(),
      paddle::platform::errors::PreconditionNotMet(
          "[Prim] For op %s, its origin output num %d is not equal to "
          "decomp output num %d ",
          op_name,
          orig_outs.size(),
          decomp_outs.size()));
  std::vector<pir::Value> new_decomp_outs(orig_outs.size());
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (orig_outs[i]) {
      PADDLE_ENFORCE_EQ(
          decomp_outs[i].size(),
          1,
          paddle::platform::errors::PreconditionNotMet(
              "[Prim] For op %s, each element of decomp output num must "
              "be 1, but num of index %d is %d ",
              op_name,
              i,
              decomp_outs[i].size()));
      new_decomp_outs[i] = decomp_outs[i][0];
    }
  }
  return new_decomp_outs;
}

std::vector<pir::Value> DecompProgram::construct_dst_vars(
    const std::string& op_name,
    const std::vector<pir::Value>& orig_outs,
    const std::vector<pir::Value>& decomp_outs,
    std::unordered_map<pir::Value, int> orig_vars_dict) {
  std::vector<pir::Value> tar_vars(src_vars_.size());
  PADDLE_ENFORCE_EQ(
      orig_outs.size(),
      decomp_outs.size(),
      paddle::platform::errors::PreconditionNotMet(
          "[Prim] For op %s, its origin output num %d is not equal to "
          "decomp output num %d ",
          op_name,
          orig_outs.size(),
          decomp_outs.size()));
  for (size_t i = 0; i < orig_outs.size(); i++) {
    if (orig_vars_dict.find(orig_outs[i]) != orig_vars_dict.end()) {
      tar_vars[orig_vars_dict[orig_outs[i]]] = decomp_outs[i];
    }
  }
  return tar_vars;
}

std::vector<pir::Value> DecompProgram::get_dst_vars() {
  if (!paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
    return src_vars_;
  } else {
    return dst_vars_;
  }
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

std::vector<std::vector<pir::Value>> call_decomp_rule(pir::Operation* op) {
  paddle::dialect::DecompInterface decomp_interface =
      op->dyn_cast<paddle::dialect::DecompInterface>();
  PADDLE_ENFORCE(decomp_interface,
                 phi::errors::InvalidArgument(
                     "[Prim] The decomp function is not registered in %s op ",
                     op->name()));
  std::vector<std::vector<pir::Value>> decomp_res = decomp_interface.Decomp(op);
  return decomp_res;
}

void DecompProgram::decomp_program() {
  std::unordered_map<pir::Value, int> orig_vars_dict;
  for (size_t i = 0; i < src_vars_.size(); i++) {
    orig_vars_dict[src_vars_[i]] = static_cast<int>(i);
  }
  std::ostringstream orig_prog_stream;
  program_->Print(orig_prog_stream);
  VLOG(4) << "[Prim] Origin program before decomp :\n"
          << orig_prog_stream.str();

  if (!paddle::prim::PrimCommonUtils::IsFwdPrimEnabled()) {
    return;
  }
  std::vector<pir::Value> tar_vars(src_vars_.size());
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
    if (enable_prim) {
      VLOG(4) << "[Prim] decomp op name " << op->name();
      check_decomp_dynamic_shape(op);
      auto& builder = *(paddle::dialect::ApiBuilder::Instance().GetBuilder());
      builder.set_insertion_point(op);
      std::vector<std::vector<pir::Value>> decomp_res = call_decomp_rule(op);
      std::vector<pir::Value> orig_outs = op->results();
      std::vector<pir::Value> standard_decomp_res =
          format_decomp_res(op->name(), orig_outs, decomp_res);
      check_decomp_outputs(op->name(), orig_outs, standard_decomp_res);
      tar_vars = construct_dst_vars(
          op->name(), orig_outs, standard_decomp_res, orig_vars_dict);

      op->ReplaceAllUsesWith(standard_decomp_res);
      bool remove_op = true;
      for (auto& item : op->results()) {
        if (item.HasOneUse()) {
          remove_op = false;
          break;
        }
      }
      if (remove_op) {
        auto op_iter = std::find(block->begin(), block->end(), *op);
        block->erase(op_iter);
      }
    }
  }
  for (size_t i = 0; i < tar_vars.size(); i++) {
    if (!tar_vars[i]) {
      tar_vars[i] = src_vars_[i];
    }
  }
  auto& builder = *(paddle::dialect::ApiBuilder::Instance().GetBuilder());
  builder.SetInsertionPointToBlockEnd(block);
  std::ostringstream decomp_prog_stream;
  program_->Print(decomp_prog_stream);
  VLOG(4) << "[Prim] New program after decomp :\n" << decomp_prog_stream.str();
  dst_vars_ = tar_vars;
  return;
}

}  // namespace paddle
