// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/config/group_tile_util.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"
#include "paddle/cinn/ir/ir_analyzer/ir_analyzer.h"

namespace cinn {

using hlir::framework::pir::trivial_fusion_detail::GetAllForIters;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ChildScheduleBlockRealizes;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ChildTensorLoads;
using hlir::framework::pir::trivial_fusion_detail::ExprSetFinderUtils::
    ScheduleBlockRealizeIsNotInit;

namespace ir {

bool GetCanApplyGridReduce(const std::vector<ir::Expr>& op_compute_bodies,
                           const std::vector<int64_t>& reduce_axis) {
  // Names of tensors that are downstream of reduce.
  // A tensor is downstream of reduce either if it is produced by a reduce, or
  // if it has data dependency on another tensor that is downstream of reduce.
  std::unordered_set<std::string> reduce_downstream_tensor_names;
  int reduce_count = 0;

  const auto IsReduceDownstream = [&](const ir::Expr& expr_block) {
    for (auto& expr_load : ChildTensorLoads(expr_block)) {
      std::string load_tensor_name = expr_load.As<ir::Load>()->name();
      if (reduce_downstream_tensor_names.count(load_tensor_name) > 0) {
        return true;
      }
    }
    return false;
  };

  const auto AddReduceDownstream = [&](const ir::Expr& expr_block) {
    auto expr_store = analyzer::GetStoreOfSBlock(expr_block);
    std::string store_tensor_name = expr_store.As<ir::Store>()->name();
    reduce_downstream_tensor_names.insert(store_tensor_name);
  };

  const auto CheckOutputHasReduceAxis = [&](const ir::Expr& body,
                                            const ir::Expr& expr_block) {
    std::vector<ir::Var> all_loop_vars = GetAllForIters(body);
    std::unordered_set<std::string> reduce_loop_vars;
    for (int64_t axis : reduce_axis) {
      reduce_loop_vars.insert(all_loop_vars[axis]->name);
    }

    std::unordered_set<std::string> reduce_iter_vars;
    auto* block = expr_block.As<ir::ScheduleBlockRealize>();
    auto& iter_vars = block->schedule_block.As<ir::ScheduleBlock>()->iter_vars;
    for (int i = 0; i < iter_vars.size(); i++) {
      ir::Var loop_var = block->iter_values[i].as_var_ref();
      if (reduce_loop_vars.count(loop_var->name) > 0) {
        reduce_iter_vars.insert(iter_vars[i]->name);
      }
    }

    // The result is true if the indices of the output tensor contain any
    // reduce iter vars.
    auto expr_store = analyzer::GetStoreOfSBlock(expr_block);
    for (auto& index_expr : expr_store.As<ir::Store>()->indices) {
      if (reduce_iter_vars.count(index_expr.as_var_ref()->name) > 0) {
        return true;
      }
    }
    return false;
  };

  for (const auto& body : op_compute_bodies) {
    ir::Expr expr_block =
        (ChildScheduleBlockRealizes * ScheduleBlockRealizeIsNotInit)
            .GetSingle(body);
    bool is_reduce = analyzer::IsReductionSBlock(expr_block);
    bool is_reduce_downstream = IsReduceDownstream(expr_block);
    bool output_has_reduce_axis = CheckOutputHasReduceAxis(body, expr_block);

    if (is_reduce) {
      ++reduce_count;
    }
    if (is_reduce_downstream || is_reduce) {
      AddReduceDownstream(expr_block);
    }

    // When a block is downstream of reduce, its output shouldn't contain
    // reduce axis. Otherwise, it broadcasts the result of reduce. If this
    // is the case, we cannot apply grid reduce.
    if (is_reduce_downstream && output_has_reduce_axis) {
      VLOG(4) << "grid reduce is prohibited by block: " << expr_block;
      return false;
    }
  }

  return reduce_count == 1;
}

}  // namespace ir
}  // namespace cinn
