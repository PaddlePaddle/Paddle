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
  ir_builder.data_->result = lower_func;
}

void IfContextNode::ExitWithContext() {
  IRContextNode::ExitWithContext();
  if (!exprs.empty()) {
    PADDLE_THROW(::common::errors::InvalidArgument(
        "Expr not be either in ThenBlock or ElseBlock in if"));
  }
  if (!true_case.defined()) {
    PADDLE_THROW(
        ::common::errors::InvalidArgument("Expr not be defined in ThenBlock"));
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

ir::LoweredFunc IRBuilderNode::GetResult() const {
  PADDLE_ENFORCE_EQ(
      result.defined(),
      true,
      ::common::errors::InvalidArgument("No result generated in IRBuilder."));
  return result;
}

void IRBuilderNode::Reset() {
  contexts.clear();
  result.Reset();
}

IRBuilder::IRBuilder() {
  cinn::common::Shared<IRBuilderNode> n(new IRBuilderNode());
  n->Reset();
  data_ = n;
}

void IRBuilder::EnterWithContext() {
  PADDLE_ENFORCE_EQ(
      data_->contexts.empty(),
      true,
      ::common::errors::InvalidArgument(
          "There are still contexts in IRBuilder that have not been fully "
          "converted. Please build a new IR with the new IRBuilder."));

  data_->result.Reset();
  std::vector<IRBuilder>* st = IRBuilderStack();
  st->push_back(*this);
}

void IRBuilder::ExitWithContext() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  PADDLE_ENFORCE_EQ(!st->empty(),
                    true,
                    ::common::errors::InvalidArgument(
                        "The IRBuilder stack must not be empty."));
  st->pop_back();
}
IRBuilder IRBuilder::CurrentIRBuilder() {
  std::vector<IRBuilder>* st = IRBuilderStack();
  PADDLE_ENFORCE_EQ(
      !st->empty(),
      true,
      ::common::errors::InvalidArgument("No IRBuilder found in the stack."));
  return st->back();
}
std::vector<IRBuilder>* IRBuilderStack() {
  thread_local std::vector<IRBuilder> stack;
  return &stack;
}
void LinkToParentContext(ir::Expr expr) {
  IRBuilder ir_builder = IRBuilder::CurrentIRBuilder();
  PADDLE_ENFORCE_GT(ir_builder.data_->contexts.size(),
                    0,
                    ::common::errors::InvalidArgument(
                        "No parent context found in IRBuilder."));
  IRContext ir_context = ir_builder.data_->contexts.back();
  ir_context.add_expr(expr);
}

}  // namespace pybind
}  // namespace cinn
