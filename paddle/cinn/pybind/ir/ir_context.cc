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

#include "paddle/cinn/pybind/ir/ir_context.h"
#include "paddle/cinn/ir/ir.h"

namespace cinn {
namespace pybind {
void IRContextNode::EnterWithContext() {
  IRBuilder::CurrentIRBuilder().data_->contexts.emplace_back(this);
}
void IRContextNode::ExitWithContext() {
  IRBuilder::CurrentIRBuilder().data_->contexts.pop_back();
}

void ScheduleBlockContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  ir::Expr schedule_block = ir::ScheduleBlock::Make(
      iter_vars, read_buffers, write_buffers, name, ir::Block::Make(exprs));

  ir::Expr schedule_block_realize =
      ir::ScheduleBlockRealize::Make(iter_values, schedule_block);
  LinkToParentContext(schedule_block_realize);
}

void ForContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  LinkToParentContext(ir::For::Make(loop_var,
                                    min,
                                    extent,
                                    ir::ForType::Serial,
                                    ir::DeviceAPI::UNK,
                                    ir::Block::Make(exprs)));
}

void LowerFuncContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  // TODO(6clc): implement Private Fields for intrinstic function, like
  // allreduce
  Expr body = ir::ScheduleBlockRealize::Make(
      {}, ir::ScheduleBlock::Make({}, {}, {}, "root", ir::Block::Make(exprs)));
  ir::LoweredFunc lower_func =
      ir::_LoweredFunc_::Make(name, args, ir::Block::Make({body}));
  IRBuilder ir_builder = IRBuilder::CurrentIRBuilder();
  ir_builder.data_->result = lower_func.operator Expr();
}

void IfContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  if (!exprs.empty()) {
    LOG(FATAL) << "Expr not be either in ThenBlock or ElseBlock in if";
  }
  if (!true_case.defined()) {
    LOG(FATAL) << "Expr not be defined in ThenBlock";
  }
  LinkToParentContext(ir::IfThenElse::Make(condition, true_case, false_case));
}

void ThenContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  IRContext for_ctx =
      IRBuilder::CurrentIRBuilder().data_->GetLastContext<IfContextNode>();
  for_ctx.data_->safe_as<IfContextNode>()->true_case = ir::Block::Make(exprs);
}

void ElseContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  IRContext for_ctx =
      IRBuilder::CurrentIRBuilder().data_->GetLastContext<IfContextNode>();
  for_ctx.data_->safe_as<IfContextNode>()->false_case = ir::Block::Make(exprs);
}

Expr IRBuilderNode::GetResult() const {
  CHECK(result.defined()) << "No result generated in IRBuilder";
  return result;
}

void IRBuilderNode::Reset() {
  contexts.clear();
  result.Reset();
}

IRBuilder::IRBuilder() {
  common::Shared<IRBuilderNode> n(new IRBuilderNode());
  n->Reset();
  data_ = n;
}

void IRBuilder::EnterWithContext() {
  CHECK(data_->contexts.empty())
      << "There are still Contexts in IRBuilder that has not been fully "
         "converted. Please build a new IR with the new IRbuilder";
  data_->result.Reset();
  std::vector<IRBuilder>* st = IRBuilderStack();
  st->push_back(*this);
}

void IRBuilder::ExitWithContext() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  CHECK(!st->empty());
  st->pop_back();
}
IRBuilder IRBuilder::CurrentIRBuilder() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  CHECK(!st->empty()) << "No IRBuilder Found";
  return st->back();
}
std::vector<IRBuilder>* IRBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}
void LinkToParentContext(ir::Expr expr) {
  IRBuilder ir_builder = IRBuilder::CurrentIRBuilder();
  if (ir_builder.data_->contexts.empty()) {
    ir_builder.data_->result = expr;
  } else {
    IRContext ir_context = ir_builder.data_->contexts.back();
    ir_context.add_expr(expr);
  }
}

}  // namespace pybind
}  // namespace cinn
