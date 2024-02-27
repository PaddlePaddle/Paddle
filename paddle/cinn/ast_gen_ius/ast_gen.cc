// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ast_gen_ius/ast_gen.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/operation.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/lang/compute.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"

PD_DECLARE_bool(cinn_new_group_scheduler);
PD_DECLARE_bool(group_schedule_tiling_first);
PD_DECLARE_bool(cinn_bucket_compile);

namespace cinn {
namespace ast_gen_ius {

ir::Expr ConvertReduceBody(ir::Expr body,
                           ir::Tensor tensor,
                           const std::vector<Expr>& axis_exprs) {
  ir::Reduce* reduce_node = body.As<ir::Reduce>();
  if (!reduce_node) {
    return ir::Store::Make(tensor, body, axis_exprs);
  }

  switch (reduce_node->reduce_type) {
    case ir::Reduce::kSum:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) + reduce_node->body, axis_exprs);
    case ir::Reduce::kMul:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) * reduce_node->body, axis_exprs);
    case ir::Reduce::kMax:
      return ir::Store::Make(
          tensor,
          ir::Max::Make(tensor(axis_exprs), reduce_node->body),
          axis_exprs);
    case ir::Reduce::kMin:
      return ir::Store::Make(
          tensor,
          ir::Min::Make(tensor(axis_exprs), reduce_node->body),
          axis_exprs);
    case ir::Reduce::kAll:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) && reduce_node->body, axis_exprs);
    case ir::Reduce::kAny:
      return ir::Store::Make(
          tensor, tensor(axis_exprs) || reduce_node->body, axis_exprs);
    default:
      CINN_NOT_IMPLEMENTED
  }
}

ir::Expr AstGen::Build(const ir::Tensor& tensor, TensorGroup* tensor_group) {
  const std::vector<ir::Var>& axis = tensor->axis();
  const std::vector<ir::Expr>& shape = tensor->shape;
  size_t axis_len = axis.size();
  CHECK_EQ(shape.size(), axis_len) << "Internal Error: Tensor has different "
                                      "shape and axis length in AstGen";
  std::vector<ir::Expr> axis_exprs;
  for (const auto& a : axis) {
    axis_exprs.push_back(a);
  }

  if (tensor->is_reduce_tensor()) {
    // Make an init Tensor for domain without reduce axis
    Expr init_value = tensor->GetReduceInitVal();
    // TODO(zhhsplendid): Clean the hardcoded "__reduce_init" string
    std::string reduce_init_name = tensor->name + "__reduce_init";
    const std::vector<Expr>& domain = tensor->domain_without_reduce_axis();
    ir::Tensor init_tensor = lang::Compute(
        domain,
        [=](const std::vector<Expr>& axis) { return init_value; },
        reduce_init_name);
    tensor_group->Insert(init_tensor);
    tensor_group->MarkShareMemBuffer(tensor, init_tensor);
    tensor_group->CtrlDepend(tensor, init_tensor);
    Expr init_body = ir::Store::Make(init_tensor, init_value, axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    std::vector<ir::Expr> iter_values;
    // reduce body and reduce init schedule block should have different objects
    // for same axis so we re-create objects
    VLOG(4) << "FLAGS_group_schedule_tiling_first = "
            << FLAGS_group_schedule_tiling_first;
    std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(axis_len);
    const std::vector<ir::Var>& reduce_axis = tensor->reduce_axis;
    const auto reduce_axis_position = [&reduce_axis,
                                       &tensor]() -> std::vector<int> {
      VLOG(4) << "start calculus reduce_axis_position: ";
      std::vector<int> res;
      auto fn_body = tensor->operation.ptr()->as<ir::ComputeOp>()->body[0];
      if (fn_body.defined() && fn_body.As<ir::Reduce>()) {
        auto& reduce_body =
            fn_body.As<ir::Reduce>()->body;  // reduce body is a tensor store.
        auto& load_indices = reduce_body.As<ir::Load>()->indices;
        int position = -1;
        for (auto& obj : load_indices) {
          position += 1;
          for (auto& reduce_var : reduce_axis) {
            if (obj.as_var_ref() == reduce_var) {
              res.push_back(position);
            }
          }
        }
        for (auto i : res) {
          VLOG(4) << "reduce axis position is " << i;
        }
        return res;
      }
    }();
    for (int i = 0; i < shape.size(); ++i) {
      if (FLAGS_group_schedule_tiling_first &&
          std::find(reduce_axis_position.begin(),
                    reduce_axis_position.end(),
                    i) != reduce_axis_position.end()) {
        // if tiling first, we need to replace the reduce axis with 0, but don't
        // deal with the non-reduce axis
        optim::ReplaceVarWithExpr(&init_body, axis[i], Expr(0));
        continue;
      }
      if (!FLAGS_group_schedule_tiling_first &&
          FLAGS_cinn_new_group_scheduler && shape[i] == Expr(1)) {
        optim::ReplaceVarWithExpr(&init_body, axis[i], Expr(0));
        continue;
      }
      block_vars.push_back(Var(Expr(0),
                               shape[i],
                               cinn::UniqName("i" + std::to_string(i)),
                               /*is_reduce = */ false));
      optim::ReplaceVarWithExpr(&init_body, axis[i], block_vars.back());
      axis_vars[i]->is_reduce_axis = false;
      if (!FLAGS_group_schedule_tiling_first && shape[i] == Expr(1)) {
        iter_values.push_back(Expr(0));
      } else {
        iter_values.push_back(axis_vars[i]);
      }
    }
    init_body = ir::ScheduleBlockRealize::Make(
        iter_values,
        ir::ScheduleBlock::Make(
            block_vars, {}, {}, reduce_init_name, init_body));

    // For the remaining reduce axis, make reduce body
    ir::Expr reduce_body =
        ConvertReduceBody(tensor->body(), tensor, axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> reduce_block_vars;
    std::vector<ir::Expr> reduce_iter_values;
    // reduce body and reduce init schedule block should have different objects
    // for same axis so we re-create objects
    std::vector<Var> reduce_axis_vars = cinn::common::GenDefaultAxis(axis_len);
    for (int i = 0; i < shape.size(); ++i) {
      if (FLAGS_group_schedule_tiling_first &&
          std::find(reduce_axis_position.begin(),
                    reduce_axis_position.end(),
                    i) != reduce_axis_position.end()) {
        // if tiling first, we need to replace the reduce axis with 0, but don't
        // deal with the non-reduce axis
        optim::ReplaceVarWithExpr(&reduce_body, axis[i], Expr(0));
        continue;
      }
      if (!FLAGS_group_schedule_tiling_first &&
          FLAGS_cinn_new_group_scheduler && shape[i] == Expr(1)) {
        optim::ReplaceVarWithExpr(&reduce_body, axis[i], Expr(0));
        continue;
      }
      reduce_block_vars.push_back(Var(Expr(0),
                                      shape[i],
                                      cinn::UniqName("i" + std::to_string(i)),
                                      /*is_reduce = */ false));
      reduce_axis_vars[i]->is_reduce_axis = false;
      if (!FLAGS_group_schedule_tiling_first && shape[i] == Expr(1)) {
        reduce_iter_values.push_back(Expr(0));
      } else {
        reduce_iter_values.push_back(axis_vars[i]);
      }
    }
    for (int i = 0; i < reduce_axis.size(); ++i) {
      int count = shape.size() + i;
      reduce_block_vars.push_back(
          Var(reduce_axis[i]->lower_bound,
              reduce_axis[i]->upper_bound,
              cinn::UniqName("i" + std::to_string(count)),
              /*is_reduce = */ true));
      ir::Var reduce_axis_var = reduce_axis[i];
      reduce_axis_var->is_reduce_axis = true;
      reduce_iter_values.push_back(reduce_axis_var);
    }

    int non_zero_axis_size = 0;
    for (int i = 0; i < axis.size(); ++i) {
      if (!FLAGS_group_schedule_tiling_first &&
          FLAGS_cinn_new_group_scheduler && shape[i] == Expr(1)) {
        continue;
      }
      optim::ReplaceVarWithExpr(
          &reduce_body, axis[i], reduce_block_vars[non_zero_axis_size]);
      ++non_zero_axis_size;
    }

    if (FLAGS_group_schedule_tiling_first) {
      non_zero_axis_size = axis.size() - reduce_axis.size();
    }
    for (int i = non_zero_axis_size; i < reduce_block_vars.size(); ++i) {
      optim::ReplaceVarWithExpr(&reduce_body,
                                reduce_axis[i - non_zero_axis_size],
                                reduce_block_vars[i]);
    }

    reduce_body = ir::ScheduleBlockRealize::Make(
        reduce_iter_values,
        ir::ScheduleBlock::Make(
            reduce_block_vars, {}, {}, tensor->name, reduce_body));
    for (int i = static_cast<int>(reduce_axis.size()) - 1; i >= 0; --i) {
      reduce_body = ir::For::Make(reduce_axis[i],
                                  reduce_axis[i]->lower_bound,
                                  reduce_axis[i]->upper_bound,
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::Host,
                                  ir::Block::Make({reduce_body}));
    }

    // Put the two parts together
    ir::Expr body = ir::Block::Make({init_body, reduce_body});
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      if (FLAGS_group_schedule_tiling_first &&
          std::find(reduce_axis_position.begin(),
                    reduce_axis_position.end(),
                    i) != reduce_axis_position.end()) {
        continue;
      }
      if (!FLAGS_group_schedule_tiling_first /*&& !FLAGS_cinn_bucket_compile*/
          && shape[i] == Expr(1)) {
        continue;
      }
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      body = ir::For::Make(
          loop_var,
          Expr(0),
          loop_extent,
          ir::ForType::Serial,
          ir::DeviceAPI::Host,
          i == static_cast<int>(axis_len) - 1 ? body : ir::Block::Make({body}));
    }
    return body;
  } else {
    ir::Expr body = ir::Store::Make(tensor, tensor->body(), axis_exprs);
    // create schedule block itervars, i0,i1...
    std::vector<ir::Var> block_vars;
    std::vector<ir::Expr> iter_values;
    std::vector<Var> axis_vars = cinn::common::GenDefaultAxis(axis_len);
    for (int i = 0; i < shape.size(); ++i) {
      block_vars.push_back(Var(
          Expr(0), shape[i], cinn::UniqName("i" + std::to_string(i)), false));
      optim::ReplaceVarWithExpr(&body, axis[i], block_vars[i]);
      axis_vars[i]->is_reduce_axis = false;
      if (!FLAGS_group_schedule_tiling_first && shape[i] == Expr(1)) {
        iter_values.push_back(Expr(0));
      } else {
        iter_values.push_back(axis_vars[i]);
      }
    }
    body = ir::ScheduleBlockRealize::Make(
        iter_values,
        ir::ScheduleBlock::Make(block_vars, {}, {}, tensor->name, body));
    for (int i = static_cast<int>(axis_len) - 1; i >= 0; --i) {
      ir::Var loop_var = axis[i];
      ir::Expr loop_extent = shape[i];
      body = ir::For::Make(loop_var,
                           Expr(0),
                           loop_extent,
                           ir::ForType::Serial,
                           ir::DeviceAPI::Host,
                           ir::Block::Make({body}));
    }
    return body;
  }
}

}  // namespace ast_gen_ius
}  // namespace cinn
