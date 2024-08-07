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

#include "paddle/cinn/pybind/ir/ir.h"
#include "paddle/cinn/pybind/ir/ir_context.h"
namespace cinn {
namespace pybind {
void TensorStore(Expr tensor, Expr value, const std::vector<Expr>& indices) {
  // TODO(6clc): Check the compatibility of data types for tensor and value
  IRContext find_sch_block =
      IRBuilder::CurrentIRBuilder()
          .data_->FindContext<ScheduleBlockContextNode>();
  if (!find_sch_block.data_.defined()) {
    IRContext sch_block(new ScheduleBlockContextNode());
    sch_block.data_->EnterWithContext();
    LinkToParentContext(ir::Store::Make(tensor, value, indices));
    sch_block.data_->ExitWithContext();
    return;
  }
  LinkToParentContext(ir::Store::Make(tensor, value, indices));
}
std::vector<Expr> AxisMap(const std::string& kinds,
                          const std::vector<Expr>& iter_expression) {
  std::vector<Expr> rets;
  PADDLE_ENFORCE_EQ(
      kinds.size(),
      iter_expression.size(),
      ::common::errors::InvalidArgument(
          "The size of kinds and iter expression in AxisMap is not equal,"
          "where kinds size:%d but iter expression size:%d.",
          kinds.size(),
          iter_expression.size()));
  int n = iter_expression.size();
  rets.reserve(n);
  for (int i = 0; i < n; i++) {
    char c = kinds.c_str()[i];

    // TODO(6clc): set bound of IterVar

    Var iter_var = ir::_Var_::Make("iter_tmp", cinn::common::Int(32));
    if (c == 'S') {
      iter_var->is_reduce_axis = false;
    } else if (c == 'R') {
      iter_var->is_reduce_axis = true;
    } else {
      PADDLE_THROW(::common::errors::InvalidArgument(
          "kind of axis setting error, must be R(Reduce) or S(Spatial)"));
    }
    rets.push_back(SetScheduleBlockIterVar(iter_var, iter_expression[i]));
  }
  return rets;
}
Var SetScheduleBlockIterVar(Var iter_var, Expr expr) {
  IRContext cur_context =
      IRBuilder::CurrentIRBuilder()
          .data_->GetLastContext<ScheduleBlockContextNode>();
  ScheduleBlockContextNode* cur_context_node =
      cur_context.As<ScheduleBlockContextNode>();
  cur_context_node->iter_vars.push_back(iter_var);
  cur_context_node->iter_values.push_back(expr);
  return iter_var.operator Expr();
}

Expr Arg(const std::string& name, Var var) {
  IRContext ctx =
      IRBuilder::CurrentIRBuilder().data_->FindContext<LowerFuncContextNode>();
  var->name = name;
  ctx.As<LowerFuncContextNode>()->args.emplace_back(var,
                                                    ir::Argument::IO::kUnknown);
  return var.operator Expr();
}

Expr Arg(const std::string& name, ir::Buffer buffer) {
  IRContext ctx =
      IRBuilder::CurrentIRBuilder().data_->FindContext<LowerFuncContextNode>();
  buffer->name = "_" + name;
  // TODO(6clc): Unify cinn compilation and runtime Type,
  //  and add a Handle type to Var
  ctx.As<LowerFuncContextNode>()->args.emplace_back(buffer,
                                                    ir::Argument::IO::kUnknown);
  return buffer.operator Expr();
}

IRContext Sequential(Expr min, Expr extent) {
  ForContextNode* for_ctx_node = new ForContextNode();
  for_ctx_node->min = min;
  for_ctx_node->extent = extent;
  for_ctx_node->loop_var = ir::_Var_::Make("v", cinn::common::Int(32));
  return IRContext(for_ctx_node);
}

}  // namespace pybind

}  // namespace cinn
