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
#include "paddle/common/enforce.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir_adaptor/translator/attribute_translator.h"
#include "paddle/fluid/ir_adaptor/translator/op_translator.h"
#include "paddle/fluid/ir_adaptor/translator/type_translator.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/pir/core/attribute.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/value.h"
#include "paddle/pir/dialect/control_flow/ir/cf_dialect.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"

namespace paddle {
namespace translator {

using ProgramDesc = ::paddle::framework::ProgramDesc;
using BlockDesc = ::paddle::framework::BlockDesc;
using VarDesc = ::paddle::framework::VarDesc;

using TCKey = TranslationContext::Key;
using TCValue = TranslationContext::Value;
using TCContainer = TranslationContext::Container;

const std::unordered_set<std::string> ProgramTranslator::no_cast_var_names = {
    "feed",
    "fetch",
};

const std::unordered_set<std::string> ProgramTranslator::unsupported_ops = {
    "conditional_block_grad",
    "while_grad",
};

static std::vector<uint64_t> GetCondOpIds(const BlockDesc& src_block,
                                          uint64_t first_id) {
  uint64_t temp_id = first_id;
  // add conditional_block
  std::vector<uint64_t> op_list = {temp_id};
  temp_id++;
  // add logical_not
  if ((temp_id < src_block.OpSize()) &&
      (src_block.Op(static_cast<int>(temp_id))->Type() == "logical_not")) {
    op_list.emplace_back(temp_id);
    temp_id++;
  }
  // add conditional_block
  if ((temp_id < src_block.OpSize()) &&
      (src_block.Op(static_cast<int>(temp_id))->Type() ==
       "conditional_block")) {
    op_list.emplace_back(temp_id);
    temp_id++;
  }
  // add cast
  if ((temp_id < src_block.OpSize()) &&
      (src_block.Op(static_cast<int>(temp_id))->Type() == "cast")) {
    op_list.emplace_back(temp_id);
    temp_id++;
  }
  // Note(zhangbo): Some output variables are input, without select_input op.
  std::vector<uint64_t> init_op_list;
  while (temp_id < src_block.OpSize()) {
    if ((src_block.Op(static_cast<int>(temp_id))->Type() == "fill_constant") ||
        (src_block.Op(static_cast<int>(temp_id))->Type() == "assign_value")) {
      init_op_list.emplace_back(temp_id);
      temp_id++;
    } else {
      break;
    }
  }
  std::vector<uint64_t> select_input_op_list;
  while (temp_id < src_block.OpSize()) {
    if (src_block.Op(static_cast<int>(temp_id))->Type() == "select_input") {
      select_input_op_list.emplace_back(temp_id);
      temp_id++;
    } else {
      break;
    }
  }

  if (select_input_op_list.size() > 0) {
    op_list.insert(op_list.end(), init_op_list.begin(), init_op_list.end());
  }
  op_list.insert(
      op_list.end(), select_input_op_list.begin(), select_input_op_list.end());

  return op_list;
}

ConditionBlockCombination::ConditionBlockCombination(
    const ::paddle::framework::BlockDesc& src_block,
    const std::vector<uint64_t>& op_ids) {
  for (auto op_id : op_ids) {
    op_list_.emplace_back(src_block.Op(static_cast<int>(op_id)));
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

std::set<std::string> ConditionBlockCombination::GetInputNamesForIfOp() const {
  std::set<std::string> input_names;
  if (op_list_.size() == 0) {
    return input_names;
  }

  if (TrueBlockId() != -1) {
    auto vars = op_list_[0]->Input("Input");
    input_names.insert(vars.begin(), vars.end());
  }

  if (FalseBlockId() != -1) {
    auto vars = op_list_[2]->Input("Input");
    input_names.insert(vars.begin(), vars.end());
  }

  return input_names;
}

// NOTE(zhangbo): Special cases need to be handled for the following:
// a,b,c = true_cond(...)
// d,e   = false_cond(...)
// f     = select_input(a,d)
// g     = select_input(b,e)
// If op output includes: f,g,c
// If op true branch output includes: a,b,c
// If op false branch output includes: d,e,c
std::tuple<std::vector<::paddle::framework::VarDesc*>,
           std::vector<std::string>,
           std::vector<std::string>>
ConditionBlockCombination::CondOutputVars() const {
  std::set<std::string> if_outputs;
  std::unordered_map<std::string, std::string> true_out_map;
  std::unordered_map<std::string, std::string> false_out_map;
  for (::paddle::framework::OpDesc* op : op_list_) {
    if (op->Type() == "conditional_block") {
      if_outputs.insert(op->Output("Out").begin(), op->Output("Out").end());
    }
    if (op->Type() == "select_input") {
      if_outputs.insert(op->Output("Out")[0]);
      if_outputs.erase(op->Input("X")[0]);
      if_outputs.erase(op->Input("X")[1]);
      false_out_map[op->Output("Out")[0]] = op->Input("X")[0];
      true_out_map[op->Output("Out")[0]] = op->Input("X")[1];
    }
  }
  std::vector<::paddle::framework::VarDesc*> if_output_vars;
  std::vector<std::string> true_outputs;
  std::vector<std::string> false_outputs;
  for (auto cond_out : if_outputs) {
    if_output_vars.emplace_back(
        op_list_[0]->Block()->FindVarRecursive(cond_out));
    if (true_out_map.find(cond_out) != true_out_map.end()) {
      true_outputs.emplace_back(true_out_map[cond_out]);
    } else {
      true_outputs.emplace_back(cond_out);
    }
    if (false_out_map.find(cond_out) != false_out_map.end()) {
      false_outputs.emplace_back(false_out_map[cond_out]);
    } else {
      false_outputs.emplace_back(cond_out);
    }
  }
  return {if_output_vars, true_outputs, false_outputs};
}

size_t ConditionBlockCombination::MainOutputSize() const {
  return std::get<0>(CondOutputVars()).size();
}

std::vector<std::string> ConditionBlockCombination::TrueBlockOutputVarNames()
    const {
  std::vector<std::string> output_names;
  for (::paddle::framework::OpDesc* op : op_list_) {
    if (op->Type() == "select_input") {
      output_names.emplace_back(op->Input("X")[1]);
    }
  }
  return output_names;
}

std::vector<::paddle::framework::OpDesc*>
ConditionBlockCombination::TrueBlockInitOps() const {
  std::vector<::paddle::framework::OpDesc*> init_ops;
  std::vector<std::string> output_names = TrueBlockOutputVarNames();
  for (::paddle::framework::OpDesc* op : op_list_) {
    if ((op->Type() == "fill_constant") || (op->Type() == "assign_value")) {
      auto out_name = op->Output("Out")[0];
      if (std::find(output_names.begin(), output_names.end(), out_name) !=
          output_names.end()) {
        init_ops.emplace_back(op);
      }
    }
  }
  return init_ops;
}

int ConditionBlockCombination::TrueBlockId() const {
  return op_list_[0]->GetBlockAttrId("sub_block");
}

std::vector<std::string> ConditionBlockCombination::FalseBlockOutputVarNames()
    const {
  std::vector<std::string> output_names;
  for (::paddle::framework::OpDesc* op : op_list_) {
    if (op->Type() == "select_input") {
      output_names.emplace_back(op->Input("X")[0]);
    }
  }
  return output_names;
}

std::vector<::paddle::framework::OpDesc*>
ConditionBlockCombination::FalseBlockInitOps() const {
  std::vector<::paddle::framework::OpDesc*> init_ops;
  std::vector<std::string> output_names = FalseBlockOutputVarNames();
  for (::paddle::framework::OpDesc* op : op_list_) {
    if ((op->Type() == "fill_constant") || (op->Type() == "assign_value")) {
      auto out_name = op->Output("Out")[0];
      if (std::find(output_names.begin(), output_names.end(), out_name) !=
          output_names.end()) {
        init_ops.emplace_back(op);
      }
    }
  }
  return init_ops;
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
      if ((op_list[id]->Type() != "select_input") &&
          (op_list[id]->Type() != "fill_constant") &&
          (op_list[id]->Type() != "assign_value")) {
        return false;
      }
    }
  }
  return true;
}

const TCValue& TranslationContext::operator[](const TCKey& key) const {
  return at(key);
}

const TCValue& TranslationContext::at(const TCKey& key) const {
  auto it = container_.find(key);
  if (it == container_.end() && parent_) {
    return parent_->at(key);
  }
  PADDLE_ENFORCE_NE(it,
                    container_.end(),
                    platform::errors::InvalidArgument(
                        "param %s should exists in TranslationContext", key));
  const auto& values = it->second;
  PADDLE_ENFORCE_NE(
      values.size(),
      0,
      platform::errors::InvalidArgument(
          "param %s should have size > 0, but get:%d", key, values.size()));
  return values.back();
}

bool TranslationContext::Has(const Key& key) const {
  return container_.find(key) != container_.end() ||
         (parent_ && parent_->Has(key));
}

size_t TranslationContext::count(const TCKey& key) const {
  auto it = container_.find(key);
  if (it == container_.end()) {
    if (parent_) return parent_->count(key);
    return 0u;
  }
  const auto& values = it->second;
  PADDLE_ENFORCE_NE(
      values.size(),
      0u,
      platform::errors::InvalidArgument(
          "param %s should have size > 0, but get:%d", key, values.size()));
  return values.size();
}

void TranslationContext::PushValue(const Key& key, const Value& value) {
  container_[key].push_back(value);
}
void TranslationContext::PopValue(const Key& key) {
  container_[key].pop_back();
}

TranslationContext* TranslationContext::CreateInnerContext() {
  sons_.emplace_back(std::make_unique<TranslationContext>(this));
  return sons_.back().get();
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
                 &param_map_,
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
                                       TranslationContext* translation_ctx,
                                       pir::Block* dst_block) {
  VLOG(8) << "=============>start to translate a block";
  PADDLE_ENFORCE(
      (src_block.OpSize() >= end_id) && (start_id <= end_id),
      platform::errors::NotFound(
          "Translation of Block needs to meet the requirements of start_id <= "
          "end_id <= block_size, but get start_id=%d, end_id=%d, block_size=%d",
          start_id,
          end_id,
          src_block.OpSize()));

  std::map<std::string, std::string> assign_output_2_input;
  for (uint64_t op_id = start_id; op_id < end_id; op_id++) {
    auto op = src_block.Op(static_cast<int>(op_id));
    VLOG(8) << "=============>start to translate a op: " << op->Type();

    PADDLE_ENFORCE_EQ(unsupported_ops.count(op->Type()),
                      0,
                      platform::errors::PreconditionNotMet(
                          "Not support translated %s op", op->Type()));

    if (op->Type() == "conditional_block") {
      TranslateIfOperation(op, translation_ctx, dst_block);
    } else if (op->Type() == "while") {
      TranslateWhileOperation(op, translation_ctx, dst_block);
    } else {
      TranslateGeneralOperation(op, translation_ctx, dst_block);
    }
  }
}

// NOTE(zhangbo): All condition_block_op will be translated as an if_op with
// only a true branch.
void ProgramTranslator::TranslateIfOperation(
    const OpDesc* op,
    TranslationContext* translation_ctx,
    pir::Block* dst_block) {
  VLOG(8) << "=============>Start to translate if op:" << op;
  auto& type_translator = TypeTranslator::instance();

  auto cond_op_cond = op->Input("Cond")[0];
  auto& cond_op_inputs = op->Input("Input");
  for (auto input_name : cond_op_inputs) {
    VLOG(6) << "[general op][conditional_block][inputs: " << input_name << "]";
    GetValueOrCreateInTop(input_name, translation_ctx);
  }
  auto& cond_op_outputs = op->Output("Out");
  std::vector<::paddle::framework::VarDesc*> cond_op_output_vars;
  for (auto out_name : cond_op_outputs) {
    cond_op_output_vars.emplace_back(op->Block()->FindVarRecursive(out_name));
  }

  std::vector<pir::Value> if_op_inputs = {
      (*translation_ctx)[cond_op_cond].value};
  pir::AttributeMap attribute_map;
  std::vector<pir::Type> if_op_output_types;
  for (auto var_desc : cond_op_output_vars) {
    IR_ENFORCE(var_desc != nullptr, "[control flow] Output should not be null");
    pir::Type translated_var_type =
        type_translator[var_desc->GetType()](ctx_, *var_desc);
    if_op_output_types.emplace_back(translated_var_type);
  }
  auto if_op_info = ctx_->GetRegisteredOpInfo(paddle::dialect::IfOp::name());
  pir::Operation* operation = pir::Operation::Create(
      if_op_inputs, attribute_map, if_op_output_types, if_op_info, 2);

  dst_block->push_back(operation);
  VLOG(4) << "[general op][conditional_block] IfOp creation end.";

  if (op->GetBlockAttrId("sub_block") != -1) {
    // Translate true branch by sub_block.
    auto& sub_block = legacy_program_->Block(op->GetBlockAttrId("sub_block"));
    pir::Region& true_region = operation->region(0);
    if (true_region.empty()) true_region.emplace_back();
    auto* true_block_context = translation_ctx->CreateInnerContext();
    TranslateBlock(sub_block,
                   0,
                   sub_block.OpSize(),
                   true_block_context,
                   &true_region.front());
    // insert yeild op to true block
    auto yeild_info = ctx_->GetRegisteredOpInfo(pir::YieldOp::name());
    std::vector<pir::Value> true_yeild_inputs;
    for (auto& out_name : cond_op_outputs) {
      true_yeild_inputs.push_back(true_block_context->at(out_name).value);
    }
    true_region.front().push_back(
        pir::Operation::Create(true_yeild_inputs, {}, {}, yeild_info));

    // NOTE(zhangbo): The if_op of PIR requires that both true and false
    // branches must exist, and the number of outputs and dtypes must be
    // consistent. Only inconsistent shape is allowed. To be compatible with the
    // old IR design, only true branches are allowed. The false branch may
    // require yeild some fake variables.
    pir::Region& false_region = operation->region(1);
    if (false_region.empty()) false_region.emplace_back();
    auto* false_block_context = translation_ctx->CreateInnerContext();
    std::vector<pir::Value> false_yeild_inputs;
    for (size_t id = 0; id < cond_op_outputs.size(); id++) {
      if (false_block_context->count(cond_op_outputs[id]) == 0) {
        auto true_type = true_yeild_inputs[id].type();
        if (true_type.isa<paddle::dialect::DenseTensorType>()) {
          InsertFullOpToBlock(&false_region.front(), true_type);
        } else {
          CreateUndefinedVariable(cond_op_outputs[id], sub_block);
        }
      }
      false_yeild_inputs.push_back(
          false_block_context->at(cond_op_outputs[id]).value);
    }
    false_region.front().push_back(
        pir::Operation::Create(false_yeild_inputs, {}, {}, yeild_info));
  }
  VLOG(4) << "[general op][conditional_block] IfOp true block translate end.";

  for (size_t i = 0; i < cond_op_output_vars.size(); i++) {
    translation_ctx->PushValue(cond_op_output_vars[i]->Name(),
                               VariableDefiningInfo(operation->result(i)));
    VLOG(4) << "[general op][conditional_block] var "
            << cond_op_output_vars[i]->Name() << " was mapped to If's " << i
            << "-th output.";
  }

  operation->Verify();
  VLOG(4) << "[general op][conditional_block] IfOp translate end.";
}

void ProgramTranslator::TranslateWhileOperation(
    const OpDesc* op,
    TranslationContext* translation_ctx,
    pir::Block* dst_block) {
  VLOG(8) << "=============>Start to translate while op:" << op;
  auto& sub_block = legacy_program_->Block(op->GetBlockAttrId("sub_block"));
  auto& inputs = op->Output("Out");
  auto& cond_var = op->Input("Condition")[0];
  std::vector<std::string> loop_vars;
  for (auto& var : inputs) {
    if (var != cond_var) {
      loop_vars.emplace_back(var);
    }
  }
  auto op_info = ctx_->GetRegisteredOpInfo(paddle::dialect::WhileOp::name());
  std::vector<pir::Value> op_inputs{
      GetValueOrCreateInTop(cond_var, translation_ctx).value};
  std::vector<pir::Type> op_outputs_type;
  auto body_block = new pir::Block();
  auto* body_block_context = translation_ctx->CreateInnerContext();

  for (auto& loop_var : loop_vars) {
    auto& tc_value = GetValueOrCreateInTop(loop_var, translation_ctx);
    auto val_type = tc_value.value.type();
    op_inputs.push_back(tc_value.value);
    op_outputs_type.push_back(val_type);
    body_block_context->PushValue(loop_var, body_block->AddArgument(val_type));
  }

  pir::Operation* while_op =
      pir::Operation::Create(op_inputs, {}, op_outputs_type, op_info, 1);
  dst_block->push_back(while_op);
  while_op->region(0).push_back(body_block);

  TranslateBlock(
      sub_block, 0, sub_block.OpSize(), body_block_context, body_block);

  auto yeild_info = ctx_->GetRegisteredOpInfo(pir::YieldOp::name());
  std::vector<pir::Value> yeild_inputs{body_block_context->at(cond_var).value};
  for (auto& loop_var : loop_vars) {
    yeild_inputs.push_back(body_block_context->at(loop_var).value);
  }
  body_block->push_back(
      pir::Operation::Create(yeild_inputs, {}, {}, yeild_info));
  for (size_t idx = 0; idx < loop_vars.size(); ++idx) {
    translation_ctx->PushValue(loop_vars[idx], while_op->result(idx));
  }

  while_op->Verify();
  VLOG(8) << "=============>end to translate while op:" << op;
}

void ProgramTranslator::TranslateGeneralOperation(
    const OpDesc* src_op,
    TranslationContext* translation_ctx,
    pir::Block* dst_block) {
  auto& op_translator = OpTranslator::instance();
  OpTranslateFn& fn = op_translator[src_op->Type()];
  if (src_op->Type() == "shadow_output") {
    if (!translation_ctx->count(src_op->Input("x")[0])) {
      return;
    }
  }
  pir::Operation* operation = fn(ctx_, translation_ctx, *src_op, dst_block);
  VLOG(10) << "[op translated][general]" << operation << "end";
}

inline pir::Operation* InsertGetParamaterOp(pir::IrContext* ctx,
                                            const VarDesc* var) {
  auto& type_translator = TypeTranslator::instance();
  std::string parameter_op_name(pir::ParameterOp::name());
  pir::OpInfo op_info = ctx->GetRegisteredOpInfo(parameter_op_name);
  std::unordered_map<std::string, pir::Attribute> op_attribute_map = {
      {"parameter_name", pir::StrAttribute::get(ctx, var->Name())},
  };

  pir::Type translated_var_type = type_translator[var->GetType()](ctx, *var);
  pir::Operation* operation = pir::Operation::Create(
      {}, op_attribute_map, {translated_var_type}, op_info);
  return operation;
}

inline pir::Operation* InsertSetParamaterOp(pir::IrContext* ctx,
                                            pir::Value defining_op_result,
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

        bool need_parameter_op = is_parameter && is_unseen_variable;
        if (need_parameter_op) {
          PADDLE_ENFORCE_NOT_NULL(
              var_desc,
              phi::errors::PreconditionNotMet(
                  "VarDesc of [%s] can not be nullptr", var_name));
          pir::Operation* op = InsertGetParamaterOp(ctx_, var_desc);
          program_->block()->push_back(op);
          param_map_.PushValue(var_name, VariableDefiningInfo(op->result(0)));
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
          pir::OpResult defining_op_result =
              param_map_[var_name].value.dyn_cast<pir::OpResult>();
          if (!defining_op_result) {
            continue;
          }

          if (param_map_[var_name].generated_by_vector) {
            InsertSliceOperationForTarget(ctx_,
                                          &param_map_,
                                          program_->block(),
                                          param_map_[var_name],
                                          var_name);
            defining_op_result =
                param_map_.at(var_name).value.dyn_cast<pir::OpResult>();
          }

          pir::Operation* op = InsertSetParamaterOp(
              ctx_, defining_op_result, parameter_name_mappings_[var_name]);

          pir::Block* block = program_->block();
          pir::Block::Iterator insert_pos = std::find(
              block->begin(), block->end(), *defining_op_result.owner());

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
  for (const auto& [var_name, value_list] : param_map_) {
    if (no_cast_var_names.count(var_name) != 0) continue;
    VLOG(10) << "[op translated][stop gradient]" << var_name;
    VarDesc* var = block.FindVarRecursive(var_name);
    if (var == nullptr) {
      continue;
    }
    for (const auto& value_info : value_list) {
      pir::OpResult value = value_info.value.dyn_cast<pir::OpResult>();
      if (!value) continue;
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
      defining_op->set_attribute(
          kAttrStopGradients, pir::ArrayAttribute::get(ctx_, stop_gradients));
    }
  }
}

const VariableDefiningInfo& ProgramTranslator::GetValueOrCreateInTop(
    const std::string& var_name, TranslationContext* translation_ctx) {
  if (translation_ctx->Has(var_name)) return translation_ctx->at(var_name);
  return CreateUndefinedVariable(var_name, legacy_program_->Block(0));
}

const VariableDefiningInfo& ProgramTranslator::CreateUndefinedVariable(
    const std::string& var_name, const BlockDesc& block) {
  VLOG(10) << "[undefined variable]" << var_name;
  auto var_desc = block.FindVarRecursive(var_name);
  pir::Builder builder(ctx_, program_->block(), program_->block()->begin());
  auto dtype = ::phi::TransToPhiDataType(var_desc->GetDataType());
  auto val = pir::OpResult(nullptr);
  if (var_desc->GetType() ==
      paddle::framework::proto::VarType::LOD_TENSOR_ARRAY) {
    val = builder.Build<dialect::CreateArrayOp>(dtype).result(0);
    VLOG(10) << "[undefined variable] " << var_name << " " << val.type();
  } else {
    auto shape = var_desc->GetShape();
    val = builder
              .Build<paddle::dialect::DataOp>(
                  var_name, shape, dtype, phi::Place())
              .out();
    VLOG(10) << "[undefined variable] " << var_name << " " << val.type();
  }
  param_map_.PushValue(var_name, val);
  return param_map_.at(var_name);
}

const VariableDefiningInfo& ProgramTranslator::InsertFullOpToBlock(
    pir::Block* insert_block, pir::Type type) {
  PADDLE_ENFORCE_EQ(
      type.isa<paddle::dialect::DenseTensorType>(),
      true,
      platform::errors::InvalidArgument(
          "only support insert FullOp for DenseTensorType, but now is %s",
          type));
  pir::Builder builder(ctx_, insert_block, insert_block->begin());
  auto tensor_type = type.dyn_cast<paddle::dialect::DenseTensorType>();
  std::vector<int64_t> shape = common::vectorize(tensor_type.dims());
  paddle::dialect::FullOp full_op = builder.Build<paddle::dialect::FullOp>(
      shape,
      0,
      paddle::dialect::TransToPhiDataType(tensor_type.dtype()),
      phi::CPUPlace());
}

void ProgramTranslator::SetIsPersisableAttributeForAllValue(
    const BlockDesc& block) {
  // Currently we set is persisable for operation that generated a value
  // connected with VarDesc
  for (const auto& [var_name, value_list] : param_map_) {
    if (no_cast_var_names.count(var_name) != 0) continue;
    VLOG(10) << "[op translated][is persisable]" << var_name;
    VarDesc* var = block.FindVarRecursive(var_name);
    if (var == nullptr) {
      continue;
    }
    for (const auto& value_info : value_list) {
      pir::OpResult value = value_info.value.dyn_cast<pir::OpResult>();
      if (!value) continue;
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
}

std::unordered_map<std::string, std::vector<pir::OpResult>>
ProgramTranslator::VarDesc2OpResult() {
  std::unordered_map<std::string, std::vector<pir::OpResult>>
      var_desc_2_opresult;
  for (const auto& [var_name, value_info_list] : param_map_) {
    for (const auto& value_info : value_info_list) {
      var_desc_2_opresult[var_name].push_back(
          value_info.value.dyn_cast<pir::OpResult>());
    }
  }
  return var_desc_2_opresult;
}

}  // namespace translator
}  // namespace paddle
