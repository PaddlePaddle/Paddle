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

#include "paddle/fluid/ir_adaptor/translator/program_translator.h"

#include <unordered_map>

#include "glog/logging.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir_adaptor/translator/attribute_translator.h"
#include "paddle/fluid/ir_adaptor/translator/op_translator.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

namespace paddle {
namespace translator {

using ProgramDesc = ::paddle::framework::ProgramDesc;
using BlockDesc = ::paddle::framework::BlockDesc;
using VarDesc = ::paddle::framework::VarDesc;

const std::unordered_set<std::string> ProgramTranslator::no_cast_var_names = {
    "feed",
    "fetch",
};

const std::unordered_set<std::string> ProgramTranslator::unsupported_ops = {
    "conditional_block_grad",
    "while",
    "while_grad",
};

static std::vector<uint64_t> GetCondOpIds(const BlockDesc& src_block,
                                          uint64_t first_id) {
  std::vector<uint64_t> op_list = {first_id};
  if (src_block.Op(first_id + 1)->Type() == "logical_not") {
    op_list.emplace_back(first_id + 1);
  }
  if (src_block.Op(first_id + 2)->Type() == "conditional_block") {
    op_list.emplace_back(first_id + 2);
  }
  if (src_block.Op(first_id + 3)->Type() == "cast") {
    op_list.emplace_back(first_id + 3);
  }
  size_t output_size = src_block.Op(first_id)->Output("Out").size();
  for (size_t i = 0; i < output_size; i++) {
    if (src_block.Op(first_id + 4 + i)->Type() == "select_input") {
      op_list.emplace_back(first_id + 4 + i);
    }
  }
  return op_list;
}

static std::vector<uint64_t> GetBackwardCondOpIds(const BlockDesc& src_block,
                                                  uint64_t first_id) {
  std::vector<uint64_t> op_list = {first_id};
  int offset = 1;
  size_t output_num = 1;
  if (src_block.Op(first_id + offset)->Type() == "conditional_block_grad") {
    output_num = src_block.Op(first_id + offset)->Output("Input@GRAD").size();
    op_list.emplace_back(first_id + offset++);
  }

  if (src_block.Op(first_id + offset)->Type() == "conditional_block_grad") {
    op_list.emplace_back(first_id + offset++);
  }

  std::cerr << "output num " << output_num << std::endl;
  size_t i = 0;
  while (i < output_num) {
    if (src_block.Op(first_id + offset)->Type() == "sum") {
      i++;
    }
    op_list.emplace_back(first_id + offset++);
  }
  return op_list;
}

ConditionBlockCombination::ConditionBlockCombination(
    const ::paddle::framework::BlockDesc& src_block,
    const std::vector<uint64_t>& op_ids) {
  for (auto op_id : op_ids) {
    op_list_.emplace_back(src_block.Op(op_id));
  }
  PADDLE_ENFORCE(Verify(op_list_),
                 platform::errors::NotFound(
                     "There are cond operators in this program that do not "
                     "meet the translation requirements. Please check the "
                     "program based on the Verify function"));
}

const std::string& ConditionBlockCombination::CondVarName() const {
  return op_list_[0]->Input("Cond")[0];
}

const std::string& ConditionBlockCombination::CastOutVarName() const {
  assert(op_list_[3]->Input("X")[0] == CondVarName());
  return op_list_[3]->Output("Out")[0];
}

size_t ConditionBlockCombination::OutputSize() const {
  return op_list_[0]->Output("Out").size();
}

std::vector<::paddle::framework::VarDesc*>
ConditionBlockCombination::OutputVars() const {
  std::vector<::paddle::framework::VarDesc*> outputs;
  if (this->OutputSize() > 0) {
    for (size_t i = 4; i < op_list_.size(); i++) {
      outputs.emplace_back(op_list_[i]->Block()->FindVarRecursive(
          op_list_[i]->Output("Out")[0]));
    }
  }
  return outputs;
}

const std::vector<std::string>&
ConditionBlockCombination::TrueBlockOutputVarNames() const {
  return op_list_[0]->Output("Out");
}

int ConditionBlockCombination::TrueBlockId() const {
  return op_list_[0]->GetBlockAttrId("sub_block");
}

std::vector<std::string> ConditionBlockCombination::FalseBlockOutputVarNames()
    const {
  if (op_list_.size() > 1) {
    return op_list_[2]->Output("Out");
  }
  return {""};
}

int ConditionBlockCombination::FalseBlockId() const {
  if (op_list_.size() > 1) {
    return op_list_[2]->GetBlockAttrId("sub_block");
  }
  return -1;
}

bool ConditionBlockCombination::Verify(
    const std::vector<::paddle::framework::OpDesc*>& op_list) {
  for (size_t id = 0; id < op_list.size(); id++) {
    if (id == 0) {
      if (op_list[id]->Type() != "conditional_block") {
        return false;
      }
      if (op_list.size() == 1 && op_list[id]->Output("Out").size() != 0) {
        return false;
      }
    } else if (id == 1) {
      if (op_list[id]->Type() != "logical_not") {
        return false;
      }
      if (op_list[id]->Input("X")[0] != op_list[id - 1]->Input("Cond")[0]) {
        return false;
      }
    } else if (id == 2) {
      if (op_list[id]->Type() != "conditional_block") {
        return false;
      }
      if (op_list[id]->Input("Cond")[0] != op_list[id - 1]->Output("Out")[0]) {
        return false;
      }
    } else if (id == 3) {
      if (op_list[id]->Type() != "cast") {
        return false;
      }
      if (op_list[id]->Input("X")[0] != op_list[0]->Input("Cond")[0]) {
        return false;
      }
    } else {
      if (op_list[id]->Type() != "select_input") {
        return false;
      }
      if (op_list[id]->Input("Mask")[0] != op_list[3]->Output("Out")[0]) {
        return false;
      }
    }
  }
  return true;
}

BackwardConditionBlockCombination::BackwardConditionBlockCombination(
    const ::paddle::framework::BlockDesc& src_block,
    const std::vector<uint64_t>& op_ids) {
  output_num_ = 0;
  for (auto op_id : op_ids) {
    if (src_block.Op(op_id)->Type() == "select_output") {
      output_num_++;
    }
    op_list_.emplace_back(src_block.Op(op_id));
  }

  std::cerr << "output num " << output_num_ << std::endl;
  PADDLE_ENFORCE(Verify(op_list_),
                 platform::errors::NotFound(
                     "There are cond operators in this program that do not "
                     "meet the translation requirements. Please check the "
                     "program based on the Verify function"));
}

const std::string& BackwardConditionBlockCombination::CondVarName() const {
  return op_list_[0]->Input("Mask")[0];
}

size_t BackwardConditionBlockCombination::OutputSize() const {
  return output_num_;
}

std::vector<::paddle::framework::VarDesc*>
BackwardConditionBlockCombination::OutputVars() const {
  std::vector<::paddle::framework::VarDesc*> outputs;
  if (this->OutputSize() > 0) {
    for (size_t i = 3; i < op_list_.size(); i++) {
      if (op_list_[i]->Type() == "sum") {
        outputs.emplace_back(op_list_[i]->Block()->FindVarRecursive(
            op_list_[i]->Output("Out")[0]));
      }
    }
  }
  return outputs;
}

const std::vector<std::string>&
BackwardConditionBlockCombination::TrueBlockOutputVarNames() const {
  return op_list_[output_num_]->Output("Input@GRAD");
}

int BackwardConditionBlockCombination::TrueBlockId() const {
  if (op_list_.size() > output_num_ + 2) {
    return op_list_[output_num_ + 1]->GetBlockAttrId("sub_block");
  }
  return -1;
}

std::vector<std::string>
BackwardConditionBlockCombination::FalseBlockOutputVarNames() const {
  if (op_list_.size() > output_num_ + 2) {
    return op_list_[output_num_ + 1]->Output("Input@GRAD");
  }
  return {""};
}

int BackwardConditionBlockCombination::FalseBlockId() const {
  return op_list_[output_num_]->GetBlockAttrId("sub_block");
}

const std::vector<::paddle::framework::OpDesc*>&
BackwardConditionBlockCombination::OpList() const {
  return op_list_;
}

bool BackwardConditionBlockCombination::Verify(
    const std::vector<::paddle::framework::OpDesc*>& op_list) {
  for (size_t id = 0; id < op_list.size(); id++) {
    if (id == 0) {
      if (op_list[id]->Type() != "select_output") {
        return false;
      }
      if (op_list.size() == 1 && op_list[id]->Output("Out").size() != 0) {
        return false;
      }
    } else if (id == 1) {
      if (op_list[id]->Type() != "conditional_block_grad") {
        return false;
      }
    } else if (id == 2) {
      if (op_list[id]->Type() != "conditional_block_grad") {
        return false;
      }
    }
  }
  return true;
}

ProgramTranslator::ProgramTranslator(const ProgramDesc* legacy_program,
                                     pir::Program* program)
    : legacy_program_(legacy_program), program_(program) {
  ctx_ = pir::IrContext::Instance();
  ctx_->GetOrRegisterDialect<pir::ControlFlowDialect>();
}

void ProgramTranslator::Translate() {
  GetParameterForSingleBlock(legacy_program_->Block(0));

  TranslateBlock(legacy_program_->Block(0),
                 0,
                 legacy_program_->Block(0).OpSize(),
                 program_->block());

  SetParameterFromSingleBlock(legacy_program_->Block(0));

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetStopGradientAttributeForAllValue(block);
  }

  for (size_t block_idx = 0; block_idx < legacy_program_->Size(); block_idx++) {
    const BlockDesc& block = legacy_program_->Block(block_idx);
    SetIsPersisableAttributeForAllValue(block);
  }
}

void ProgramTranslator::TranslateBlock(const BlockDesc& src_block,
                                       uint64_t start_id,
                                       uint64_t end_id,
                                       pir::Block* dest_block,
                                       bool for_cond_block) {
  VLOG(8) << "=============>start to translate a block";
  PADDLE_ENFORCE(
      (src_block.OpSize() >= end_id) && (start_id <= end_id),
      platform::errors::NotFound(
          "Translation of Block needs to meet the requirements of start_id <= "
          "end_id <= block_size, but get start_id=%d, end_id=%d, block_size=%d",
          start_id,
          end_id,
          src_block.OpSize()));

  std::unordered_map<uint64_t, bool> translate_completed;
  for (uint64_t op_id = start_id; op_id < end_id; op_id++) {
    if (translate_completed.count(op_id) && translate_completed.at(op_id)) {
      continue;
    }
    auto op = src_block.Op(op_id);
    VLOG(8) << "=============>start to translate a op: " << op->Type();

    PADDLE_ENFORCE_EQ(unsupported_ops.count(op->Type()),
                      0,
                      platform::errors::PreconditionNotMet(
                          "Not support translated %s op", op->Type()));

    if (op->Type() == "conditional_block") {
      std::vector<const OpDesc*> cond_op_list = {op};
      std::vector<uint64_t> cond_op_ids = GetCondOpIds(src_block, op_id);
      ConditionBlockCombination cond_op_combination(src_block, cond_op_ids);
      pir::Operation* if_op =
          TranslateCondIfOperation(cond_op_combination, dest_block);
      for (auto cond_id : cond_op_ids) {
        translate_completed[cond_id] = true;
      }

      foward_condition_block_map_[cond_op_combination.CastOutVarName()] =
          cond_op_combination;
      VLOG(10) << "[op translated][conditional_block]" << if_op;
    } else if (op->Type() == "select_output") {
      std::cerr << "backward if op" << std::endl;
      std::vector<uint64_t> cond_op_ids =
          GetBackwardCondOpIds(src_block, op_id);
      BackwardConditionBlockCombination backward_cond_op_combination(
          src_block, cond_op_ids);
      pir::Operation* if_op = TranslateBackwardCondIfOperation(
          backward_cond_op_combination, dest_block);
      for (auto cond_id : cond_op_ids) {
        translate_completed[cond_id] = true;
      }
      VLOG(10) << "[op translated][conditional_block_grad]" << if_op;
    } else {
      TranslateGeneralOperation(op, dest_block);
      translate_completed[op_id] = true;
    }
  }
  // NOTE(zhangbo): If conditional_block operator has output, the cf.yeild
  // operator needs to be inserted
  if (for_cond_block) {
    std::vector<pir::Value> yeild_inputs;
    for (size_t id = end_id; id < src_block.OpSize(); id++) {
      PADDLE_ENFORCE(
          src_block.Op(id)->Type() == "assign",
          "The operator at the end of the sub block needs to be assign");
      yeild_inputs.emplace_back(
          param_map_[src_block.Op(id)->Input("X")[0]].value);
    }
    pir::AttributeMap attribute_map;
    auto yeild_info = ctx_->GetRegisteredOpInfo(pir::YieldOp::name());
    pir::Operation* yeild_op =
        pir::Operation::Create(yeild_inputs, attribute_map, {}, yeild_info);
    dest_block->push_back(yeild_op);
  }
}

pir::Operation* ProgramTranslator::TranslateCondIfOperation(
    const ConditionBlockCombination& cond_ops, pir::Block* dest_block) {
  auto& type_translator = TypeTranslator::instance();
  auto op_info = ctx_->GetRegisteredOpInfo(paddle::dialect::IfOp::name());
  std::vector<pir::Value> op_inputs = {
      param_map_[cond_ops.CondVarName()].value};

  // NOTE(zhangbo): Now paddle::dialect::IfOp has 0 attribute
  pir::AttributeMap attribute_map;

  std::vector<pir::Type> op_output_types;
  std::vector<::paddle::framework::VarDesc*> output_vardescs =
      cond_ops.OutputVars();
  for (auto var_desc : output_vardescs) {
    IR_ENFORCE(var_desc != nullptr, "[control flow] Output should not be null");
    pir::Type translated_var_type =
        type_translator[var_desc->GetType()](ctx_, *var_desc);
    op_output_types.emplace_back(translated_var_type);
  }
  VLOG(4) << "[general op][conditional_block] IfOp preparation end.";

  pir::Operation* operation = pir::Operation::Create(
      op_inputs, attribute_map, op_output_types, op_info, 2);

  for (size_t i = 0; i < output_vardescs.size(); i++) {
    std::cerr << "forward name " << output_vardescs[i]->Name() << std::endl;
    param_map_[output_vardescs[i]->Name()] =
        VariableDefiningInfo(operation->result(i));
  }

  dest_block->push_back(operation);
  VLOG(4) << "[general op][conditional_block] IfOp creation end.";

  if (cond_ops.TrueBlockId() != -1) {
    const BlockDesc& true_sub_block =
        legacy_program_->Block(cond_ops.TrueBlockId());
    pir::Region& true_region = operation->region(0);
    if (true_region.empty()) true_region.emplace_back();
    TranslateBlock(true_sub_block,
                   0,
                   true_sub_block.OpSize() - cond_ops.OutputSize(),
                   true_region.front(),
                   true);
  }
  VLOG(4) << "[general op][conditional_block] IfOp true block translate end.";

  if (cond_ops.FalseBlockId() != -1) {
    const BlockDesc& false_sub_block =
        legacy_program_->Block(cond_ops.FalseBlockId());
    pir::Region& false_region = operation->region(1);
    if (false_region.empty()) false_region.emplace_back();
    TranslateBlock(false_sub_block,
                   0,
                   false_sub_block.OpSize() - cond_ops.OutputSize(),
                   false_region.front(),
                   true);
  }
  VLOG(4) << "[general op][conditional_block] IfOp false block translate end.";
  VLOG(4) << "[general op][conditional_block] IfOp translate end.";
  return operation;
}

pir::Operation* ProgramTranslator::TranslateBackwardCondIfOperation(
    const BackwardConditionBlockCombination& cond_ops, pir::Block* dest_block) {
  auto& type_translator = TypeTranslator::instance();
  auto op_info = ctx_->GetRegisteredOpInfo(paddle::dialect::IfOp::name());

  std::cerr << "found vond var " << cond_ops.CondVarName() << "\t"
            << foward_condition_block_map_.count(cond_ops.CondVarName())
            << std::endl;
  auto& foward_cond_ops =
      foward_condition_block_map_.at(cond_ops.CondVarName());

  std::vector<pir::Value> op_inputs = {
      param_map_[foward_cond_ops.CondVarName()].value};

  // proccess select output
  for (size_t i = 0; i < cond_ops.OutputSize(); ++i) {
    auto op_desc = cond_ops.OpList()[i];
    assert(op_desc->Type() == "select_output");
    std::cerr << "selec output in " << op_desc->Input("X")[0] << "\t"
              << param_map_.count(op_desc->Input("X")[0]) << std::endl;
    param_map_[op_desc->Output("Out")[0]] = param_map_[op_desc->Input("X")[0]];
    param_map_[op_desc->Output("Out")[1]] = param_map_[op_desc->Input("X")[0]];
  }

  // NOTE(zhangbo): Now paddle::dialect::IfOp has 0 attribute
  pir::AttributeMap attribute_map;

  std::vector<pir::Type> op_output_types;
  std::vector<::paddle::framework::VarDesc*> output_vardescs =
      cond_ops.OutputVars();
  for (auto var_desc : output_vardescs) {
    IR_ENFORCE(var_desc != nullptr, "[control flow] Output should not be null");
    pir::Type translated_var_type =
        type_translator[var_desc->GetType()](ctx_, *var_desc);
    op_output_types.emplace_back(translated_var_type);
  }
  VLOG(4) << "[general op][conditional_block_grad] IfOp preparation end.";

  pir::Operation* operation = pir::Operation::Create(
      op_inputs, attribute_map, op_output_types, op_info, 2);

  std::unordered_map<std::string, pir::Value> true_branch_map;

  dest_block->push_back(operation);
  VLOG(4) << "[general op][conditional_block_grad] IfOp creation end.";

  if (cond_ops.TrueBlockId() != -1) {
    const BlockDesc& true_sub_block =
        legacy_program_->Block(cond_ops.TrueBlockId());
    pir::Region& true_region = operation->region(0);
    if (true_region.empty()) true_region.emplace_back();
    // update start here
    std::cerr << "process block id " << cond_ops.TrueBlockId() << std::endl;
    TranslateBlock(
        true_sub_block, 0, true_sub_block.OpSize(), true_region.front(), false);
    // insert yied op here
    auto cond_output_var_name = cond_ops.OutputVars();
    std::vector<pir::Value> yeild_inputs;
    for (auto& name : cond_output_var_name) {
      std::cerr << "name " << name->Name() << "\t"
                << param_map_.count(name->Name()) << std::endl;
      yeild_inputs.emplace_back(param_map_[name->Name()].value);
    }

    auto true_branch_output_var_name = cond_ops.TrueBlockOutputVarNames();

    for (size_t i = 0; i < true_branch_output_var_name.size(); ++i) {
      std::cerr << "true map " << cond_output_var_name[i]->Name() << "\t"
                << true_branch_output_var_name[i] << std::endl;
      // true_branch_map[true_branch_output_var_name[i] ] = param_map_[
      // cond_output_var_name[i]->Name() ].value;
      true_branch_map[true_branch_output_var_name[i]] = operation->result(0);
    }

    pir::AttributeMap attribute_map;
    auto yeild_info = ctx_->GetRegisteredOpInfo(pir::YieldOp::name());
    pir::Operation* yeild_op =
        pir::Operation::Create(yeild_inputs, attribute_map, {}, yeild_info);
    true_region.front()->push_back(yeild_op);
  }
  VLOG(4)
      << "[general op][conditional_block_grad] IfOp true block translate end.";

  std::unordered_set<std::string> false_branch_output_var_name;
  if (cond_ops.FalseBlockId() != -1) {
    const BlockDesc& false_sub_block =
        legacy_program_->Block(cond_ops.FalseBlockId());
    pir::Region& false_region = operation->region(1);
    if (false_region.empty()) false_region.emplace_back();
    // update start here
    std::cerr << "process false block id " << cond_ops.FalseBlockId()
              << std::endl;
    TranslateBlock(false_sub_block,
                   0,
                   false_sub_block.OpSize(),
                   false_region.front(),
                   false);
    auto cond_output_var_name = cond_ops.OutputVars();
    std::vector<pir::Value> yeild_inputs;
    for (auto& name : cond_output_var_name) {
      std::cerr << "name " << name->Name() << "\t"
                << param_map_.count(name->Name()) << std::endl;
      yeild_inputs.emplace_back(param_map_[name->Name()].value);
    }

    auto false_output_var_name = cond_ops.FalseBlockOutputVarNames();

    for (auto& name : false_output_var_name) {
      false_branch_output_var_name.insert(name);
    }

    pir::AttributeMap attribute_map;
    auto yeild_info = ctx_->GetRegisteredOpInfo(pir::YieldOp::name());
    pir::Operation* yeild_op =
        pir::Operation::Create(yeild_inputs, attribute_map, {}, yeild_info);
    false_region.front()->push_back(yeild_op);
  }

  for (size_t i = 0; i < output_vardescs.size(); i++) {
    std::cerr << "var name " << output_vardescs[i]->Name() << std::endl;
    param_map_[output_vardescs[i]->Name()] =
        VariableDefiningInfo(operation->result(i));
  }

  // proess next ops

  auto op_list = cond_ops.OpList();
  for (size_t i = cond_ops.OutputSize() + 2; i < op_list.size(); ++i) {
    if (op_list[i]->Type() == "sum") {
      if (op_list[i]->Input("X").size() == 2) {
        std::cerr << "skip sum !!" << std::endl;
        continue;
      }

      // add sum op
      std::cerr << "process sum" << std::endl;
      auto input_names = op_list[i]->Input("X");
      std::vector<pir::Value> sum_input;
      for (size_t i = 0; i < input_names.size(); ++i) {
        std::cerr << "input name " << input_names[i] << std::endl;
        if (true_branch_map.count(input_names[i])) {
          std::cerr << "true" << std::endl;
          sum_input.push_back(true_branch_map.at(input_names[i]));
        } else if (false_branch_output_var_name.count(input_names[i])) {
          std::cerr << "falter" << std::endl;
          continue;
        } else {
          assert(param_map_.count(input_names[i]));

          sum_input.push_back(param_map_.at(input_names[i]).value);
        }
      }
      std::cerr << "build here\n";

      pir::Builder builder(ctx_, dest_block);

      auto combine_op = builder.Build<pir::CombineOp>(sum_input);
      auto add_n =
          builder.Build<paddle::dialect::AddNWithKernelOp>(combine_op.out());

      param_map_[op_list[i]->Output("Out")[0]] = add_n->result(0);
    } else {
      TranslateGeneralOperation(op_list[i], dest_block);
    }
  }
  VLOG(4)
      << "[general op][conditional_block_grad] IfOp false block translate end.";
  VLOG(4) << "[general op][conditional_block_grad] IfOp translate end.";
  return operation;
}

void ProgramTranslator::TranslateGeneralOperation(const OpDesc* src_op,
                                                  pir::Block* dest_block) {
  auto& op_translator = OpTranslator::instance();
  OpTranslateFn& fn = op_translator[src_op->Type()];
  if (src_op->Type() == "shadow_output") {
    if (!param_map_.count(src_op->Input("x")[0])) {
      return;
    }
  }
  pir::Operation* operation = fn(ctx_, &param_map_, *src_op, dest_block);
  VLOG(10) << "[op translated][special] [" << operation->name() << "] end";
}

inline pir::Operation* InsertGetParamaterOp(pir::IrContext* ctx,
                                            const VarDesc* var) {
  auto& type_translator = TypeTranslator::instance();
  std::string get_parameter_op_name(pir::GetParameterOp::name());
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(get_parameter_op_name);
  std::unordered_map<std::string, pir::Attribute> op_attribute_map = {
      {"parameter_name", pir::StrAttribute::get(ctx, var->Name())},
  };

  pir::Type translated_var_type = type_translator[var->GetType()](ctx, *var);
  pir::Operation* operation = pir::Operation::Create(
      {}, op_attribute_map, {translated_var_type}, op_info);
  return operation;
}

inline pir::Operation* InsertSetParamaterOp(pir::IrContext* ctx,
                                            pir::OpResult defining_op_result,
                                            const VarDesc* var) {
  std::string set_parameter_op_name(pir::SetParameterOp::name());
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(set_parameter_op_name);
  std::unordered_map<std::string, pir::Attribute> op_attribute_map = {
      {"parameter_name", pir::StrAttribute::get(ctx, var->Name())},
  };

  pir::Operation* operation = pir::Operation::Create(
      {defining_op_result}, op_attribute_map, {}, op_info);
  return operation;
}

void ProgramTranslator::GetParameterForSingleBlock(const BlockDesc& block) {
  for (auto& var : block.AllVars()) {
    if (!var->Persistable()) continue;
    if (param_map_.count(var->Name()) != 0) continue;
    if (no_cast_var_names.count(var->Name()) != 0) continue;

    parameter_name_mappings_[var->Name()] = var;
  }

  std::unordered_set<std::string> inner_defining_variables;

  for (auto op_desc : block.AllOps()) {
    for (const auto& n : op_desc->Inputs()) {
      const auto& input_var_names = n.second;
      for (const auto& var_name : input_var_names) {
        if (no_cast_var_names.count(var_name) != 0) continue;
        VarDesc* var_desc = nullptr;

        bool is_parameter = (parameter_name_mappings_.find(var_name) !=
                             parameter_name_mappings_.end());
        is_parameter &= (parameter_visited_.count(var_name) == 0);
        if (is_parameter) {
          var_desc = parameter_name_mappings_[var_name];
        }
        bool is_unseen_variable =
            (inner_defining_variables.count(var_name) == 0);
        if (is_unseen_variable) {
          var_desc = block.FindVarRecursive(var_name);
        }

        bool need_get_parameter_op = is_parameter && is_unseen_variable;
        if (need_get_parameter_op) {
          PADDLE_ENFORCE_NOT_NULL(
              var_desc,
              phi::errors::PreconditionNotMet(
                  "VarDesc of [%s] can not be nullptr", var_name));
          pir::Operation* op = InsertGetParamaterOp(ctx_, var_desc);
          program_->block()->push_back(op);
          param_map_[var_name] = VariableDefiningInfo(op->result(0));
          VLOG(10) << "[op translated][get parameter]" << var_name;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
          inner_defining_variables.insert(var_name);
        }
      }
    }
    for (const auto& n : op_desc->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        inner_defining_variables.insert(var_name);
      }
    }
  }
}

void ProgramTranslator::InsertOperationToSingleBlock(const BlockDesc& block) {
  auto& op_translator = OpTranslator::instance();
  for (auto op : block.AllOps()) {
    OpTranslateFn& fn = op_translator[op->Type()];
    if (op->Type() == "shadow_output") {
      if (!param_map_.count(op->Input("x")[0])) {
        continue;
      }
    }
    pir::Operation* operation = fn(ctx_, &param_map_, *op, program_->block());
    VLOG(10) << "[op translated][special]" << operation;
  }
}

void ProgramTranslator::SetParameterFromSingleBlock(const BlockDesc& block) {
  const auto& ops = block.AllOps();
  for (auto op_desc = ops.rbegin(); op_desc != ops.rend(); op_desc++) {
    if ((*op_desc)->Type() == "data") {
      continue;
    }

    const auto& input_var_names = (*op_desc)->InputArgumentNames();
    std::unordered_set<std::string> set_input_var_names(input_var_names.begin(),
                                                        input_var_names.end());

    for (const auto& n : (*op_desc)->Outputs()) {
      const auto& output_var_names = n.second;
      for (const auto& var_name : output_var_names) {
        bool need_set_parameter_op = (parameter_name_mappings_.find(var_name) !=
                                      parameter_name_mappings_.end());
        need_set_parameter_op &= (parameter_visited_.count(var_name) == 0);
        need_set_parameter_op &= (param_map_.count(var_name) != 0);
        need_set_parameter_op &= (!set_input_var_names.count(var_name));
        if (need_set_parameter_op) {
          pir::OpResult defining_op_result = param_map_[var_name].value;
          if (!defining_op_result) {
            continue;
          }

          if (param_map_[var_name].generated_by_vector) {
            InsertSliceOperationForTarget(ctx_,
                                          &param_map_,
                                          program_->block(),
                                          param_map_[var_name],
                                          var_name);
            defining_op_result = param_map_.at(var_name).value;
          }

          pir::Operation* op = InsertSetParamaterOp(
              ctx_, defining_op_result, parameter_name_mappings_[var_name]);

          pir::Block* block = program_->block();
          pir::Block::Iterator insert_pos = std::find(
              block->begin(), block->end(), defining_op_result.owner());

          IR_ENFORCE(
              insert_pos != block->end(),
              "Parameter %s must have corresponding its defining operation",
              var_name);
          insert_pos++;

          block->insert(insert_pos, op);
          VLOG(10) << "[op translated][set parameter]" << var_name;

          program_->SetParameter(var_name, nullptr);
          parameter_visited_.insert(var_name);
        }
      }
    }
  }
}

void ProgramTranslator::SetStopGradientAttributeForAllValue(
    const BlockDesc& block) {
  // Currently we set stop gradient for operation that generated a value
  // connected with VarDesc
  for (const auto& [var_name, value_info] : param_map_) {
    if (no_cast_var_names.count(var_name) != 0) continue;
    VLOG(10) << "[op translated][stop gradient]" << var_name;
    VarDesc* var = block.FindVarRecursive(var_name);
    if (var == nullptr) {
      continue;
    }
    pir::OpResult value = value_info.value;
    if (!value) {
      PADDLE_THROW(phi::errors::PreconditionNotMet(
          "Value of [%s] can not ber None", var_name));
    }
    auto* defining_op = value.owner();
    PADDLE_ENFORCE_NOT_NULL(
        defining_op,
        phi::errors::PreconditionNotMet(
            "Defining operator of [%s] can not be nullptr", var_name));
    VLOG(8) << "[op translated][stop gradient]" << var_name
            << " from: " << defining_op->name();
    std::vector<pir::Attribute> stop_gradients;
    if (defining_op->HasAttribute(kAttrStopGradients)) {
      stop_gradients = defining_op->attribute(kAttrStopGradients)
                           .dyn_cast<pir::ArrayAttribute>()
                           .AsVector();
    } else {
      stop_gradients = std::vector<pir::Attribute>(
          defining_op->num_results(), pir::BoolAttribute::get(ctx_, false));
    }
    stop_gradients[value.index()] =
        pir::BoolAttribute::get(ctx_, var->StopGradient());
    defining_op->set_attribute(kAttrStopGradients,
                               pir::ArrayAttribute::get(ctx_, stop_gradients));
  }
}

void ProgramTranslator::SetIsPersisableAttributeForAllValue(
    const BlockDesc& block) {
  // Currently we set is persisable for operation that generated a value
  // connected with VarDesc
  for (const auto& [var_name, value_info] : param_map_) {
    if (no_cast_var_names.count(var_name) != 0) continue;
    VLOG(10) << "[op translated][is persisable]" << var_name;
    VarDesc* var = block.FindVarRecursive(var_name);
    if (var == nullptr) {
      continue;
    }
    pir::OpResult value = value_info.value;
    if (!value) {
      PADDLE_THROW(phi::errors::PreconditionNotMet(
          "Value of [%s] can not ber None", var_name));
    }
    auto* defining_op = value.owner();
    PADDLE_ENFORCE_NOT_NULL(
        defining_op,
        phi::errors::PreconditionNotMet(
            "Defining operator of [%s] can not be nullptr", var_name));
    VLOG(8) << "[op translated][is persisable]" << var_name
            << " from: " << defining_op->name();
    std::vector<pir::Attribute> is_persisable;
    if (defining_op->HasAttribute(kAttrIsPersisable)) {
      is_persisable = defining_op->attribute(kAttrIsPersisable)
                          .dyn_cast<pir::ArrayAttribute>()
                          .AsVector();
    } else {
      is_persisable = std::vector<pir::Attribute>(
          defining_op->num_results(), pir::BoolAttribute::get(ctx_, false));
    }
    is_persisable[value.index()] =
        pir::BoolAttribute::get(ctx_, var->Persistable());
    defining_op->set_attribute(kAttrIsPersisable,
                               pir::ArrayAttribute::get(ctx_, is_persisable));
  }
}

}  // namespace translator
}  // namespace paddle
