// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/auto_schedule/analysis/analyze_ir.h"

#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_set>

#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/ir/utils/ir_nodes_collector.h"
#include "paddle/cinn/lang/lower.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/optim/transform_gpu_forloop.h"

namespace cinn {
namespace auto_schedule {

std::vector<ir::Var> IndicesToVars(const std::vector<ir::Expr>& indices) {
  std::vector<ir::Var> result;
  for (const ir::Expr& e : indices) {
    // Whether we have to convert other types, like const numbers to Var?
    if (e.As<ir::_Var_>() != nullptr) {
      ir::Expr copy_e = ir::ir_utils::IRCopy(e);
      ir::_Var_* var_ref = copy_e.As<ir::_Var_>();
      result.emplace_back(ir::Var(var_ref));
    }
  }
  return result;
}

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  ir::ir_utils::CollectIRNodesWithoutTensor(
      sche_block->body, [&](const Expr* x) {
        const ir::Load* load_expr = x->As<ir::Load>();
        if (load_expr != nullptr) {
          const ir::Tensor t = load_expr->tensor.as_tensor_ref();
          sche_block->read_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(load_expr->indices)));
          return false;
        }
        const ir::Store* store_expr = x->As<ir::Store>();
        if (store_expr != nullptr) {
          const ir::Tensor t = store_expr->tensor.as_tensor_ref();
          sche_block->write_buffers.emplace_back(
              ir::BufferRange(t->buffer, IndicesToVars(store_expr->indices)));
          return false;
        }
        return false;
      });
}

bool ContainsNodeType(ir::Expr expr,
                      const std::unordered_set<ir::IrNodeTy>& node_types) {
  std::set<ir::Expr> collection =
      ir::ir_utils::CollectIRNodesWithoutTensor(expr, [&](const Expr* x) {
        return node_types.find(x->node_type()) != node_types.end();
      });
  return !collection.empty();
}

std::unordered_set<std::string> GetOutputNamesFromLoweredFunc(
    const std::vector<ir::LoweredFunc>& lowered_funcs) {
  std::unordered_set<std::string> result;
  for (const ir::LoweredFunc& func : lowered_funcs) {
    for (const ir::Argument& arg : func->args) {
      if (arg.is_output()) {
        result.insert(arg.name());
      }
    }
  }
  return result;
}

bool NeedsMultiLevelTiling(const ir::ScheduleBlockRealize& sche_block_realize) {
  const ir::ScheduleBlock* sche_block =
      sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  if (sche_block->write_buffers.size() != 1 ||
      sche_block->read_buffers.empty()) {
    return false;
  }
  const ir::Expr& write_buffer =
      sche_block->write_buffers[0].As<ir::_BufferRange_>()->buffer;

  // Enumerate each read region, get the number of schedule block iter vars
  // which  are not used to index the read region
  int total_unused_iter_vars = 0;

  for (const ir::Expr& read_buffer_expr : sche_block->read_buffers) {
    const ir::_BufferRange_* read_buffer =
        read_buffer_expr.As<ir::_BufferRange_>();
    // Skip the reduction buffer
    if (read_buffer->buffer == write_buffer) {
      continue;
    }
    // Collect the vars in schedule block that are used to index the read region
    std::unordered_set<std::string> vars_index_read;
    for (const Var& range : read_buffer->ranges) {
      vars_index_read.insert(range->name);
    }
    // Check the block iter vars are not used to index the read region
    int n_unused_block_vars = 0;
    for (const ir::Var& block_iter_var : sche_block->iter_vars) {
      if (!block_iter_var->is_reduce_axis) {
        bool iter_var_in_read = false;
        for (const std::string& var : vars_index_read) {
          if (var == block_iter_var->name) {
            iter_var_in_read = true;
            break;
          }
        }
        if (!iter_var_in_read) {
          ++n_unused_block_vars;
        }
      }
    }
    total_unused_iter_vars += n_unused_block_vars;
  }

  return total_unused_iter_vars >= 1;
}

ir::LoweredFunc UpdateFuncWithNewBody(const common::Target& target,
                                      const ir::LoweredFunc& old_func,
                                      ir::Expr& body) {  // NOLINT
  ir::ModuleExpr mod_expr(std::vector<ir::Expr>({body}));
  ir::IRSchedule ir_sch(mod_expr);

  // temp_bufs may be deleted during auto tuning (such as auto inline),
  // we have to check from old temp bufs and set them as local buffer.
  for (const ir::Buffer& buf : old_func->temp_bufs) {
    const std::string& buf_name = buf->name;
    std::vector<ir::Expr> all_block_realizes = ir_sch.GetAllBlocks();
    for (ir::Expr& e : all_block_realizes) {
      const ir::ScheduleBlockRealize* sche_block_realize =
          e.As<ir::ScheduleBlockRealize>();
      const std::string& sche_name =
          sche_block_realize->schedule_block.As<ir::ScheduleBlock>()->name;
      if (buf_name == "_" + sche_name) {
        VLOG(6) << "Set local buffer for temp buffer " << buf_name;
        ir_sch.SetBuffer(e, "local", true);
        break;
      }
    }
  }

  ir::Expr updated_body = ir_sch.GetModule().GetExprs()[0];
#ifdef CINN_WITH_CUDA
  optim::OptimizeExprGPU(&updated_body);
#endif

  // Get new temp bufs by analyzing.
  std::vector<ir::Buffer> new_temp_bufs =
      lang::GetTempBuffers(old_func->args, updated_body);
  ir::LoweredFunc new_func = ir::_LoweredFunc_::Make(
      old_func->name, old_func->args, updated_body, new_temp_bufs);
#ifdef CINN_WITH_CUDA
  if (target == common::DefaultNVGPUTarget()) {
    new_func->PrepareCudaAxisInfoFromBody();
  }
#endif
  new_func =
      optim::Optimize(Expr(new_func), target, false).as_lowered_func_ref();
  new_func->PrepareBufferCastExprs(/*with_expr_gen_tensor = */ false);

  return new_func;
}

}  // namespace auto_schedule
}  // namespace cinn
