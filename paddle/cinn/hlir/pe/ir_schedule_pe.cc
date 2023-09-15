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

#include "paddle/cinn/hlir/pe/ir_schedule_pe.h"

#include <absl/container/flat_hash_map.h>
#include <isl/cpp.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <utility>

#include "paddle/cinn/common/cas.h"
#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/hlir/pe/load_x86_params.h"
#include "paddle/cinn/hlir/pe/schedule.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/poly/isl_utils.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace hlir {
namespace pe {

void IRElementwiseSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                           const std::vector<int> &output_shape,
                           const common::Target &target) {
  VLOG(3) << "Before IRElementwiseSchedule, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
  if (target == common::DefaultNVGPUTarget()) {
    auto blocks = ir_sch.GetAllBlocks();
    std::vector<ir::Expr> loops = ir_sch.GetLoops(blocks[0]);
    ir::Expr loop = ir_sch.Fuse(loops);

    auto size = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    if (size <= target.max_num_threads()) {
      ir_sch.Bind(loop, "threadIdx.x");
    } else {
      auto splited = ir_sch.Split(loop, {-1, target.max_num_threads()});
      ir_sch.Bind(splited[0], "blockIdx.x");
      ir_sch.Bind(splited[1], "threadIdx.x");
    }
  } else {
    // IRScheduleInjectiveCPU(ir_sch, output_shape, target, false);
    auto blocks = ir_sch.GetAllBlocks();
    ir_sch.FlattenLoops(ir_sch.GetLoops(blocks[0]), true);
  }
  VLOG(3) << "After IRElementwiseSchedule, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRInjectiveSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                         const std::vector<int> &output_shape,
                         const common::Target &target) {
  VLOG(3) << "Before IRInjectiveSchedule, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
  if (target == common::DefaultNVGPUTarget()) {
    auto blocks = ir_sch.GetAllBlocks();
    std::vector<ir::Expr> loops = ir_sch.GetLoops(blocks[0]);
    ir::Expr loop = ir_sch.Fuse(loops);

    auto size = std::accumulate(
        output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    if (size <= target.max_num_threads()) {
      ir_sch.Bind(loop, "threadIdx.x");
    } else {
      auto splited = ir_sch.Split(loop, {-1, target.max_num_threads()});
      ir_sch.Bind(splited[0], "blockIdx.x");
      ir_sch.Bind(splited[1], "threadIdx.x");
    }
  } else {
    // IRScheduleInjectiveCPU(ir_sch, output_shape, target, false);
    auto blocks = ir_sch.GetAllBlocks();
    ir_sch.FlattenLoops(ir_sch.GetLoops(blocks[0]), false);
  }
  VLOG(3) << "After IRInjectiveSchedule, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRScheduleInjectiveCPU(ir::IRSchedule &ir_sch,  // NOLINT
                            const std::vector<int> &output_shape,
                            const common::Target &target,
                            bool vectorizable) {
  VLOG(3) << "Begin IRScheduleInjectiveCPU"
          << ir_sch.GetModule().GetExprs().at(0);
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops = ir_sch.GetLoops(all_blocks[0]);
  int dims = output_shape.size();
  int factor = GetBasicFactor(GetTensor(all_blocks[0])->type(), target);
  auto fused = loops[0];
  if (dims >= 5) {
    CHECK_GE(loops.size(), 3U);
    fused = ir_sch.Fuse({loops[0], loops[1], loops[2]});
    dims = dims - 2;
  } else if (dims >= 3) {
    CHECK_GE(loops.size(), 2U);
    fused = ir_sch.Fuse({loops[0], loops[1]});
    dims = dims - 1;
  }
  // This part needs to be fixed. @Haoze
  /*   ir_sch.Parallel(fused);
    if (vectorizable) {
      auto all_blocks = ir_sch.GetAllBlocks();
      auto loops      = ir_sch.GetLoops(all_blocks[0]);
      int last_shape  = ir::GetLoopExtent(loops.back());
      factor          = GetVectorizeFactor(last_shape, factor);
      auto splited    = ir_sch.Split(loops.back(), {-1, factor});
      ir_sch.Vectorize(splited[1], factor);
      if (dims == 1) {
        ir_sch.Parallel(splited[0]);
      }
    } */
  VLOG(3) << "After IRScheduleInjectiveCPU, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleInjective(ir::IRSchedule &ir_sch,  // NOLINT
                             const std::vector<int> &output_shape,
                             const common::Target &target) {
  VLOG(3) << "Begin IRCudaScheduleInjective ";
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops = ir_sch.GetLoops(all_blocks[0]);
  auto fused = ir_sch.Fuse(loops);

  int num_thread = target.max_num_threads();
  int vector_width = 1;
  int prod_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  if (prod_size > num_thread) {
    auto splited = ir_sch.Split(fused, {-1, num_thread});
    ir_sch.Bind(splited[0], "blockIdx.x");
    ir_sch.Bind(splited[1], "threadIdx.x");
  } else {
    ir_sch.Bind(fused, "threadIdx.x");
  }
  VLOG(3) << "After IRCudaScheduleInjective, new ir is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

std::vector<common::CINNValue> IRCudaScheduleMatMul(
    const common::CINNValuePack &arg_pack,
    const std::vector<int> &output_shape,
    const common::Target &target) {
  if (target.arch == Target::Arch::X86) {
    CINN_NOT_IMPLEMENTED
  }
  std::vector<Expr> vec_ast;
  for (int i = 0; i < arg_pack.size(); i++) {
    if (arg_pack[i].is_expr()) {
      Expr temp = arg_pack[i];
      vec_ast.emplace_back(temp);
    }
  }
  CHECK(!vec_ast.empty());
  ir::ModuleExpr mod_expr(vec_ast);
  ir::IRSchedule ir_sch(mod_expr);
  ir_sch.MergeExprs();
  // Generally, there are 2 ScheduleBlocks in the lowered function,
  // the first is for reduce_init and the second is the real compute block,
  // here we use loops of the first block to Bind GPU index in top spatial axies
  auto init_block = ir_sch.GetAllBlocks().front();
  VLOG(3) << "Matmul lowered expr:\n" << ir_sch.GetModule().GetExprs().front();

  int prod_size = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  if (prod_size > 1) {
    int num_thread = target.max_num_threads();
    auto loops = ir_sch.GetLoops(init_block);
    if (loops.size() == 1) {
      if (ir::GetLoopExtent(loops[0]) > num_thread) {
        auto splited = ir_sch.Split(loops[0], {-1, num_thread});
        ir_sch.Bind(splited[0], "blockIdx.x");
        ir_sch.Bind(splited[1], "threadIdx.x");
      } else {
        ir_sch.Bind(loops[0], "threadIdx.x");
      }
    } else {
      if (ir::GetLoopExtent(loops[1]) > num_thread) {
        ir_sch.Split(loops[1], {-1, num_thread});
        init_block = ir_sch.GetAllBlocks().front();
        ir_sch.Fuse(init_block, {0, 1});
        init_block = ir_sch.GetAllBlocks().front();
        loops = ir_sch.GetLoops(init_block);
      }
      ir_sch.Bind(loops[0], "blockIdx.x");
      ir_sch.Bind(loops[1], "threadIdx.x");
    }
  }

  return {common::CINNValue(ir_sch.GetModule().GetExprs().at(0))};
}

void IRCudaScheduleMul(ir::IRSchedule &ir_sch,  // NOLINT
                       const std::vector<int> &output_shape,
                       const common::Target &target) {
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops = ir_sch.GetLoops(all_blocks.back());
  CHECK_GE(loops.size(), 2U);
  auto splited = ir_sch.Split(loops[1], {-1, 2});
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks.back());
  ir_sch.Bind(loops[0], "blockIdx.x");
  ir_sch.Bind(loops[1], "threadIdx.x");
}

void IRMulScheduleCPU(ir::IRSchedule &ir_sch,  // NOLINT
                      const std::vector<int> &reduce_first_shape,
                      const common::Target &target) {
  ir_sch.MergeExprs();
  auto all_blocks = ir_sch.GetAllBlocks();
  CHECK_EQ(all_blocks.size(), 4U);
  auto loops = ir_sch.GetLoops(all_blocks[1]);
  int loop_size = loops.size();
  // ir_sch.Reorder({loops[loop_size-1], loops[loop_size-2]});

  if (reduce_first_shape.back() > 1) {
    all_blocks = ir_sch.GetAllBlocks();
    loops = ir_sch.GetLoops(all_blocks[3]);
    ir_sch.Unroll(loops.back());
  }
}

void IRCudaSplitSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                         const std::vector<std::vector<int>> &output_shapes,
                         int axis,
                         const common::Target &target) {
  VLOG(3) << "In IRCudaSplitSchedule, Before schedule expr is : "
          << ir_sch.GetModule().GetExprs().at(0);
  ir_sch.MergeExprs();
  // if all output are with same shape
  bool with_same_shape = true;
  for (int idx = 1; idx < output_shapes.size(); ++idx) {
    if (output_shapes[0] != output_shapes[idx]) {
      with_same_shape = false;
      break;
    }
  }

  // collect block names
  auto get_block_name = [](ir::Expr expr) {
    CHECK(expr.As<ir::ScheduleBlockRealize>());
    CHECK(expr.As<ir::ScheduleBlockRealize>()
              ->schedule_block.As<ir::ScheduleBlock>());
    return expr.As<ir::ScheduleBlockRealize>()
        ->schedule_block.As<ir::ScheduleBlock>()
        ->name;
  };
  std::vector<std::string> block_names;
  auto blocks = ir_sch.GetAllBlocks();
  for (auto &block : blocks) {
    block_names.push_back(get_block_name(block));
  }
  // if output with same shape.
  if (with_same_shape && target == common::DefaultNVGPUTarget()) {
    // flat loops.
    {
      auto tsize = std::accumulate(output_shapes[0].begin(),
                                   output_shapes[0].end(),
                                   1,
                                   std::multiplies<int>());
      for (auto &block_name : block_names) {
        ir_sch.FlattenLoops(ir_sch.GetLoops(block_name), false);

        if (tsize > target.max_num_threads()) {
          // split [-1, 256]
          auto splited = ir_sch.Split(ir_sch.GetLoops(block_name)[0],
                                      {-1, target.max_num_threads() / 4});
          ir_sch.Bind(splited[0], "blockIdx.x");
          ir_sch.Bind(splited[1], "threadIdx.x");
        } else {
          auto splited =
              ir_sch.Split(ir_sch.GetLoops(block_name)[0], {1, tsize});
          ir_sch.Bind(splited[0], "blockIdx.x");
          ir_sch.Bind(splited[1], "threadIdx.x");
        }
      }
    }
    // do simple compute at.
    {
      for (int idx = 1; idx < block_names.size(); ++idx) {
        auto master_loops = ir_sch.GetLoops(block_names[0]);
        ir_sch.SimpleComputeAt(ir_sch.GetBlock(block_names[idx]),
                               master_loops[1]);
      }
    }
  } else if (target == common::DefaultNVGPUTarget()) {
    // flat loops.
    {
      for (int idx = 0; idx < block_names.size(); ++idx) {
        ir_sch.FlattenLoops(ir_sch.GetLoops(block_names[idx]), false);
        auto first_loop = ir_sch.GetLoops(block_names[idx])[0];
        CHECK(first_loop.As<ir::For>());
        auto tsize = first_loop.As<ir::For>()->extent.as_int32();
        if (tsize > target.max_num_threads()) {
          // split [-1, 256]
          auto splited = ir_sch.Split(ir_sch.GetLoops(block_names[idx])[0],
                                      {-1, target.max_num_threads() / 4});
          ir_sch.Bind(splited[0], "blockIdx.x");
          ir_sch.Bind(splited[1], "threadIdx.x");
        } else {
          auto splited =
              ir_sch.Split(ir_sch.GetLoops(block_names[idx])[0], {1, tsize});
          ir_sch.Bind(splited[0], "blockIdx.x");
          ir_sch.Bind(splited[1], "threadIdx.x");
        }
      }
    }
  } else {
    {
      for (auto &block_name : block_names) {
        ir_sch.FlattenLoops(ir_sch.GetLoops(block_name), false);
      }
    }
  }
  VLOG(3) << "In IRCudaSplitSchedule, After schedule expr is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleReduce(ir::IRSchedule &ir_sch,  // NOLINT
                          ir::Tensor output,
                          int last_dimension_num,
                          const common::Target &target) {
  VLOG(3) << "Before IRCudaScheduleReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
  int parallel_thread_num = 1;
  auto &output_shape = output->shape;
  for (int idx = output_shape.size() - 1;
       idx >= static_cast<int>(output_shape.size()) - last_dimension_num;
       --idx) {
    parallel_thread_num *= output_shape[idx].as_int32();
  }

  int index = ir_sch.GetLoops(output->name + "__reduce_init").size() -
              last_dimension_num;
  for (int idx = output_shape.size() - last_dimension_num;
       idx < static_cast<int>(output_shape.size()) - 1;
       ++idx) {
    auto loops = ir_sch.GetLoops(output->name);
    ir_sch.Fuse({loops[index], loops[index + 1]});
  }

  int max_block_size = target.max_num_threads();
  if (parallel_thread_num > max_block_size) {
    auto loops = ir_sch.GetLoops(output->name);
    CHECK_GE(loops.size(), index + 1);
    for (int idx = max_block_size; idx > 0; --idx) {
      if (parallel_thread_num % idx == 0) {
        auto nloops = ir_sch.Split(loops[index], {-1, idx});
        ir_sch.Bind(nloops.back(), "threadIdx.x");
        break;
      }
      CHECK_GT(idx, 1);
    }
    ++index;
  } else {
    auto loops = ir_sch.GetLoops(output->name);
    CHECK_GE(loops.size(), index + 1);
    ir_sch.Bind(loops[index], "threadIdx.x");
  }

  for (int idx = 0; idx < index - 1; ++idx) {
    auto loops = ir_sch.GetLoops(output->name);
    CHECK_GT(loops.size(), 2U);
    if (loops.size() > 2) ir_sch.Fuse({loops[0], loops[1]});
  }

  if (index > 0) {
    auto loops = ir_sch.GetLoops(output->name);
    ir_sch.Bind(loops[0], "blockIdx.x");
  }
  VLOG(3) << "After IRCudaScheduleReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleBlockReduceInternal(ir::IRSchedule &ir_sch,  // NOLINT
                                       ir::Tensor tmp_out,
                                       ir::Tensor out,
                                       const common::Target &target) {
  VLOG(3) << "Before IRCudaScheduleBlockReduceInternal : "
          << ir_sch.GetModule().GetExprs().at(0);
  int fuse_times = ir_sch.GetLoops(tmp_out->name).size() - 2;
  for (int idx = 0; idx < fuse_times; ++idx) {
    for (auto &tensor : {tmp_out, out}) {
      auto loops = ir_sch.GetLoops(tensor->name);
      CHECK_GE(loops.size(), 2U);
      ir_sch.Fuse({loops[0], loops[1]});
    }
  }

  // as out shape size = [1], insert for in ast tree.
  if (tmp_out->shape.size() == 1) {
    CHECK_EQ(out->shape[0], Expr(1));

    // block and root
    auto out_block = ir_sch.GetBlock(out->name);
    auto root_block = ir_sch.GetRootBlock(out_block);

    CHECK(out_block->as<ir::ScheduleBlockRealize>());
    CHECK(out_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create var
    auto var = ir::Var(ir::Expr(0), ir::Expr(1), common::UniqName("i"));
    out_block->as<ir::ScheduleBlockRealize>()->iter_values.push_back(var);
    out_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->iter_vars.push_back(var);

    CHECK(root_block->as<ir::ScheduleBlockRealize>());
    CHECK(root_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create for and block node
    auto for_node = ir::For::Make(var,
                                  Expr(0),
                                  Expr(1),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::UNK,
                                  ir::Block::Make({out_block}));
    auto block_node =
        ir::Block::Make({root_block->as<ir::ScheduleBlockRealize>()
                             ->schedule_block->as<ir::ScheduleBlock>()
                             ->body->as<ir::Block>()
                             ->stmts[0],
                         for_node});

    root_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->body = block_node;

    for (auto &tensor : {tmp_out, out}) {
      auto loops = ir_sch.GetLoops(tensor->name);
      ir_sch.Split(loops[0], {-1, ir::GetLoopExtent(loops[0])});
    }
  }

  auto loops_tmp_out = ir_sch.GetLoops(tmp_out->name);
  auto loops_out = ir_sch.GetLoops(out->name);
  if (loops_tmp_out.size() == 1) {
    ir_sch.Bind(loops_tmp_out[0], "threadIdx.x");
    ir_sch.Bind(loops_out[0], "threadIdx.x");
  } else {
    ir_sch.Bind(loops_tmp_out[0], "blockIdx.x");
    ir_sch.Bind(loops_tmp_out[1], "threadIdx.x");

    if (loops_out.size() == 1) {
      ir_sch.Split(loops_out[0], {-1, 1});
    }
    loops_out = ir_sch.GetLoops(out->name);
    ir_sch.Bind(loops_out[0], "blockIdx.x");
    ir_sch.Bind(loops_out[1], "threadIdx.x");
  }

  for (auto &tensor : {tmp_out}) {
    auto block = ir_sch.GetBlock(tensor->name);
    ir_sch.SetBuffer(block, "local", true);
  }

  VLOG(3) << "After IRCudaScheduleBlockReduceInternal : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleBlockReduce(ir::IRSchedule &ir_sch,  // NOLINT
                               ir::Tensor reduce_tmp_out,
                               ir::Tensor tmp_out,
                               ir::Tensor out,
                               const common::Target &target) {
  VLOG(3) << "Before IRCudaScheduleBlockReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
  int tmp_put_shape_size_without_reduce = 0;
  for (auto i : tmp_out->shape) {
    CHECK(i.is_constant());
    if (i.as_int32() != 1) tmp_put_shape_size_without_reduce++;
  }
  tmp_put_shape_size_without_reduce--;
  // fuse last parallel dimension
  int reduce_temp_out_shape_size = 0;
  for (auto i : reduce_tmp_out->shape) {
    CHECK(i.is_constant());
    if (i.as_int32() != 1) reduce_temp_out_shape_size++;
  }

  int tmp_out_shape_size = tmp_put_shape_size_without_reduce + 1;
  for (int idx = 0; idx < reduce_temp_out_shape_size - tmp_out_shape_size;
       ++idx) {
    auto loops = ir_sch.GetLoops(reduce_tmp_out->name);
    int reduce_axis = reduce_tmp_out->reduce_axis.size();
    if (loops.size() >= tmp_put_shape_size_without_reduce + 2 + reduce_axis)
      ir_sch.Fuse({loops[tmp_put_shape_size_without_reduce],
                   loops[tmp_put_shape_size_without_reduce + 1]});
  }

  // fuse parallel dimension
  for (int idx = 0; idx < tmp_put_shape_size_without_reduce - 1; ++idx) {
    for (auto &tensor : {reduce_tmp_out, tmp_out, out}) {
      auto loops = ir_sch.GetLoops(tensor->name);
      int reduce_axis = tensor->reduce_axis.size();
      if (loops.size() >= 2 + reduce_axis) {
        ir_sch.Fuse({loops[0], loops[1]});
      }
    }
  }

  // Special handling when keepdim = True in reduce stage 1. When keepdim =
  // True, shape size may not be equal to 1. But we still need to split the
  // loops, otherwise there will be a problem of data read and write conflict.
  int numel = std::accumulate(
      tmp_out->shape.begin(),
      tmp_out->shape.end(),
      1,
      [](const int &num, const ir::Expr &e) { return num * e.as_int32(); });
  if (tmp_out->shape.size() == 1 ||
      (numel == tmp_out->shape.back().as_int32())) {
    CHECK_EQ(out->shape[0], Expr(1));

    // block and root
    auto out_block = ir_sch.GetBlock(out->name);
    auto root_block = ir_sch.GetRootBlock(out_block);

    CHECK(out_block->as<ir::ScheduleBlockRealize>());
    CHECK(out_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create var
    auto var = ir::Var(ir::Expr(0), ir::Expr(1), cinn::UniqName("i"));
    out_block->as<ir::ScheduleBlockRealize>()->iter_values.push_back(var);
    out_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->iter_vars.push_back(var);

    CHECK(root_block->as<ir::ScheduleBlockRealize>());
    CHECK(root_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create for and block node
    auto for_node = ir::For::Make(var,
                                  Expr(0),
                                  Expr(1),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::UNK,
                                  ir::Block::Make({out_block}));
    auto block_node =
        ir::Block::Make({root_block->as<ir::ScheduleBlockRealize>()
                             ->schedule_block->as<ir::ScheduleBlock>()
                             ->body->as<ir::Block>()
                             ->stmts[0],
                         for_node});

    root_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->body = block_node;

    for (auto &tensor : {reduce_tmp_out, tmp_out, out}) {
      auto loops = ir_sch.GetLoops(tensor->name);
      if (loops.empty()) continue;
      ir_sch.Split(loops[0], {-1, ir::GetLoopExtent(loops[0])});
    }
  }

  // bind block and thread for reduce.
  // as outer loop range should be eqaul, get loop size.
  auto b_loop = ir::GetLoopExtent(ir_sch.GetLoops(out->name)[0]);
  // reduce_tmp_out
  {
    auto loops = ir_sch.GetLoops(reduce_tmp_out->name);
    if (loops.size() <= 2U) {
      if (ir_sch.GetLoops(tmp_out->name).size() == 1) {
        ir_sch.Split(loops[0], {b_loop, -1});
      }
      loops = ir_sch.GetLoops(reduce_tmp_out->name);
    }
    ir_sch.Bind(loops[0], "blockIdx.x");
    ir_sch.Bind(loops[1], "threadIdx.x");
  }
  // tmp_out
  {
    auto loops = ir_sch.GetLoops(tmp_out->name);
    if (loops.size() < 2U) {
      ir_sch.Split(loops.back(), {b_loop, -1});
      loops = ir_sch.GetLoops(tmp_out->name);
    }

    ir_sch.Bind(loops[0], "blockIdx.x");
    ir_sch.Bind(loops[1], "threadIdx.x");
  }
  // out
  {
    auto loops = ir_sch.GetLoops(out->name);
    if (loops.size() < 2U) {
      ir_sch.Split(loops.back(), {-1, 1});
      loops = ir_sch.GetLoops(out->name);
    }
    ir_sch.Bind(loops[0], "blockIdx.x");
    ir_sch.Bind(loops[1], "threadIdx.x");
  }

  for (auto &tensor : {reduce_tmp_out, tmp_out}) {
    auto block = ir_sch.GetBlock(tensor->name);
    ir_sch.SetBuffer(block, "local", true);
  }

  VLOG(3) << "After IRCudaScheduleBlockReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleBlockShuffleReduce(ir::IRSchedule &ir_sch,  // NOLINT
                                      ir::Tensor reshape,
                                      ir::Tensor internal,
                                      ir::Tensor reduce_out,
                                      const common::Target &target) {
  VLOG(3) << "Before IRCudaScheduleBlockShuffleReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
  // reshape compute inline
  {
    // simplify reshape index
    auto hand_write_simplify = [](std::vector<ir::Expr> loops, ir::Expr block) {
      // check exist select.
      auto find_select = ir::CollectIRNodesInOrder(
          block, [&](const Expr *x) { return x->As<ir::Select>(); });
      if (find_select.size() > 0) {
        return;
      }

      auto schedule_realize = block.As<ir::ScheduleBlockRealize>();
      auto schedule_block = block.As<ir::ScheduleBlockRealize>()
                                ->schedule_block.As<ir::ScheduleBlock>();

      int stride = 1;
      std::unordered_map<std::string, ir::Expr> var_strides;
      for (int idx = loops.size() - 1; idx > 0; --idx) {
        stride = stride * GetLoopExtent(loops[idx]);

        auto var = loops[idx - 1].As<ir::For>()->loop_var;
        var_strides[var->name] = ir::Expr(stride);
      }

      ir::Expr index = ir::Expr(schedule_block->iter_vars.back());
      for (int idx = 0; idx < schedule_block->iter_vars.size() - 1; ++idx) {
        auto var = schedule_realize->iter_values[idx].as_var();
        if (!var) {
          continue;
        }

        if (!var_strides.count(var->name)) {
          continue;
        }

        auto stride = var_strides.find(var->name)->second;
        index = index + ir::Expr(schedule_block->iter_vars[idx]) * stride;
      }

      auto exprs = ir::CollectIRNodesInOrder(
          block, [&](const Expr *x) { return x->As<ir::Load>(); });
      CHECK_EQ(exprs.size(), 1);
      auto load = exprs.front().As<ir::Load>();
      load->indices = {index};
    };
    hand_write_simplify(ir_sch.GetLoops(reshape->name),
                        ir_sch.GetBlock(reshape->name));
    auto block = ir_sch.GetBlock(reshape->name);
    ir_sch.ComputeInline(block);
    VLOG(4) << "After simplify reshape index : "
            << ir_sch.GetModule().GetExprs().at(0);
  }

  // internal bind shared
  {
    auto block = ir_sch.GetBlock(internal->name);
    ir_sch.SetBuffer(block, "shared");
  }

  //
  auto get_loop_index = [&internal](ir::Expr inner_loop, ir::Expr block) {
    auto loop_var = inner_loop.As<ir::For>()->loop_var;
    auto schedule_realize = block.As<ir::ScheduleBlockRealize>();
    auto schedule_block = block.As<ir::ScheduleBlockRealize>()
                              ->schedule_block.As<ir::ScheduleBlock>();
    CHECK_EQ(schedule_realize->iter_values.size(),
             schedule_block->iter_vars.size());

    ir::Var var_name;
    for (int idx = 0; idx < schedule_block->iter_vars.size(); ++idx) {
      if (!schedule_realize->iter_values[idx].as_var()) {
        continue;
      }
      if (schedule_realize->iter_values[idx].as_var()->name != loop_var->name) {
        continue;
      }

      var_name = schedule_block->iter_vars[idx];
      break;
    }

    auto exprs = ir::CollectIRNodesInOrder(
        block, [&](const Expr *x) { return x->As<ir::Load>(); });
    for (auto expr : exprs) {
      auto load = expr.As<ir::Load>();
      auto t = load->tensor.as_tensor_ref();
      if (t->name != internal->name) {
        continue;
      }

      int index_var_count = 0;
      for (int idx = 0; idx < load->indices.size(); ++idx) {
        if (!load->indices[idx].is_var()) {
          continue;
        }

        if (load->indices[idx].as_var()->name == var_name->name) {
          break;
        }

        ++index_var_count;
      }

      // remove dimension range = 1.
      int loop_var_count = 0;
      for (int idx = 0; idx < index_var_count; ++idx) {
        if (internal->shape[idx].as_int32() > 1) {
          ++loop_var_count;
        }
      }
      return loop_var_count;
    }
    LOG(FATAL) << "Can't find var in tensor indeces!";
  };
  auto loop_var_count = get_loop_index(ir_sch.GetLoops(reduce_out->name).back(),
                                       ir_sch.GetBlock(reduce_out->name));
  // fuse loop to bind gpu block.x
  if (loop_var_count > 1) {
    auto internal_loops = ir_sch.GetLoops(internal->name);
    std::vector<ir::Expr> fuse_internal_loops(
        internal_loops.begin(), internal_loops.begin() + loop_var_count);
    ir_sch.Fuse(fuse_internal_loops);

    auto reduce_out_loops = ir_sch.GetLoops(reduce_out->name);
    std::vector<ir::Expr> fuse_reduce_out_loops(
        reduce_out_loops.begin(), reduce_out_loops.begin() + loop_var_count);
    ir_sch.Fuse(fuse_reduce_out_loops);
  }

  VLOG(4) << "After fuse loop for blockIdx.x : "
          << ir_sch.GetModule().GetExprs().at(0);
  // fuse reduce tail to bind gpu thread.
  if (ir_sch.GetLoops(reduce_out->name + "__reduce_init").size() >
      (loop_var_count ? 2 : 1)) {
    int start_index = loop_var_count == 0 ? 0 : 1;
    // first reduce step:
    // [block.x, thread.y, tail] or [thread.y, tail]
    auto internal_loops = ir_sch.GetLoops(internal->name + "__reduce_init");
    std::vector<ir::Expr> fuse_internal_loops(
        internal_loops.begin() + start_index + 1, internal_loops.end());
    ir_sch.Fuse(fuse_internal_loops);

    // second reduce step:
    // [block.x, tail] or [tail]
    auto reduce_out_loops = ir_sch.GetLoops(reduce_out->name + "__reduce_init");
    std::vector<ir::Expr> fuse_reduce_out_loops(
        reduce_out_loops.begin() + start_index, reduce_out_loops.end());
    ir_sch.Fuse(fuse_reduce_out_loops);
  }

  VLOG(4) << "After fuse tail loop for threadIdx.x : "
          << ir_sch.GetModule().GetExprs().at(0);
  // split reduce loop to bind thread.y
  {
    if (loop_var_count > 0) {
      auto reduce_out_loops =
          ir_sch.GetLoops(reduce_out->name + "__reduce_init");
      ir_sch.Split(reduce_out_loops[1], {1, -1});
    } else {
      auto reduce_out_loops =
          ir_sch.GetLoops(reduce_out->name + "__reduce_init");
      ir_sch.Split(reduce_out_loops[0], {1, -1});
    }
  }

  std::vector<int> axis_in_nroder;
  // split internal tail to bind thread
  {
    auto start_index = loop_var_count == 0 ? 0 : 1;
    auto i_loops = ir_sch.GetLoops(internal->name + "__reduce_init");
    auto r_loops = ir_sch.GetLoops(reduce_out->name + "__reduce_init");
    // bind blockIdx.x
    if (loop_var_count) {
      ir_sch.Bind(i_loops[0], "blockIdx.x");
      i_loops = ir_sch.GetLoops(internal->name + "__reduce_init");

      ir_sch.Bind(r_loops[0], "blockIdx.x");
      r_loops = ir_sch.GetLoops(reduce_out->name + "__reduce_init");

      axis_in_nroder.push_back(0);
    }
    // bind threadIdx.y
    {
      ir_sch.Bind(i_loops[start_index], "threadIdx.y");
      i_loops = ir_sch.GetLoops(internal->name + "__reduce_init");

      ir_sch.Bind(r_loops[start_index], "threadIdx.y");
      r_loops = ir_sch.GetLoops(reduce_out->name + "__reduce_init");

      axis_in_nroder.push_back(start_index);
    }

    auto bind_thread = [&](int tail) {
      if (GetLoopExtent(i_loops[start_index + 1]) > tail) {
        ir_sch.Split(i_loops[start_index + 1], {-1, tail});
        i_loops = ir_sch.GetLoops(internal->name + "__reduce_init");

        ir_sch.Split(r_loops[start_index + 1], {-1, tail});
        r_loops = ir_sch.GetLoops(reduce_out->name + "__reduce_init");

        ir_sch.Bind(i_loops[start_index + 1], "blockIdx.y");
        ir_sch.Bind(r_loops[start_index + 1], "blockIdx.y");

        ir_sch.Bind(i_loops[start_index + 2], "threadIdx.x");
        ir_sch.Bind(r_loops[start_index + 2], "threadIdx.x");

        axis_in_nroder.insert(axis_in_nroder.end() - 1, start_index + 1);
        axis_in_nroder.insert(axis_in_nroder.end() - 1, start_index + 2);
      } else {
        ir_sch.Bind(i_loops[start_index + 1], "threadIdx.x");
        ir_sch.Bind(r_loops[start_index + 1], "threadIdx.x");

        axis_in_nroder.insert(axis_in_nroder.end() - 1, start_index + 1);
      }
    };
    // split and bind blockIdx.y/threadIdx.x
    if (GetLoopExtent(i_loops[start_index]) > 32) {
      bind_thread(8);
    } else if (GetLoopExtent(i_loops[start_index]) > 16) {
      bind_thread(16);
    } else if (GetLoopExtent(i_loops[start_index]) > 4) {
      bind_thread(32);
    } else {
      bind_thread(64);
    }
  }
  VLOG(4) << "After split tail loop for threadIdx.x : "
          << ir_sch.GetModule().GetExprs().at(0);
  // do reorder
  {
    ir_sch.Reorder(internal->name + "__reduce_init", axis_in_nroder);
    ir_sch.Reorder(reduce_out->name + "__reduce_init", axis_in_nroder);
  }
  // unroll last dim
  {
    auto i_loops = ir_sch.GetLoops(internal->name);
    if (ir_sch.GetLoops(internal->name + "__reduce_init").size() <
            i_loops.size() &&
        GetLoopExtent(i_loops.back()) <= 64) {
      ir_sch.Unroll(i_loops.back());
    }

    auto r_loops = ir_sch.GetLoops(reduce_out->name);
    if (ir_sch.GetLoops(reduce_out->name + "__reduce_init").size() <
        r_loops.size()) {
      ir_sch.Unroll(r_loops.back());
    }
  }
  VLOG(3) << "After IRCudaScheduleBlockShuffleReduce : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaTwoStepReduceSchedule(ir::IRSchedule &ir_sch,  // NOLINT
                                 ir::Tensor reshape,
                                 ir::Tensor internal,
                                 ir::Tensor tmp_out,
                                 ir::Tensor out,
                                 const common::Target &target) {
  VLOG(3) << "Before IRCudaTwoStepReduceSchedule : "
          << ir_sch.GetModule().GetExprs().at(0);
  // fuse axis
  int fuse_times =
      ir_sch.GetLoops(internal->name).size() - internal->reduce_axis.size() - 2;
  for (int idx = 0; idx < fuse_times; ++idx) {
    for (auto &tensor : {internal, tmp_out, out}) {
      auto block = ir_sch.GetBlock(tensor->name);
      auto loops = ir_sch.GetLoops(block);
      int reduce_axis = tensor->reduce_axis.size();
      ir_sch.Fuse({loops[0], loops[1]});
    }
  }

  if (ir_sch.GetLoops(tmp_out->name).size() == 1) {
    // block and root
    auto out_block = ir_sch.GetBlock(out->name);
    auto root_block = ir_sch.GetRootBlock(out_block);

    CHECK(out_block->as<ir::ScheduleBlockRealize>());
    CHECK(out_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create var
    // auto var = ir::Var(ir::Expr(0), ir::Expr(1), "i_0");
    auto var = ir::Var(ir::Expr(0), ir::Expr(1), cinn::UniqName("i"));
    out_block->as<ir::ScheduleBlockRealize>()->iter_values.push_back(var);
    out_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->iter_vars.push_back(var);

    CHECK(root_block->as<ir::ScheduleBlockRealize>());
    CHECK(root_block->as<ir::ScheduleBlockRealize>()
              ->schedule_block->as<ir::ScheduleBlock>());

    // create for and block node
    auto for_node = ir::For::Make(var,
                                  Expr(0),
                                  Expr(1),
                                  ir::ForType::Serial,
                                  ir::DeviceAPI::UNK,
                                  ir::Block::Make({out_block}));

    auto block_node =
        ir::Block::Make({root_block->as<ir::ScheduleBlockRealize>()
                             ->schedule_block->as<ir::ScheduleBlock>()
                             ->body->as<ir::Block>()
                             ->stmts[0],
                         root_block->as<ir::ScheduleBlockRealize>()
                             ->schedule_block->as<ir::ScheduleBlock>()
                             ->body->as<ir::Block>()
                             ->stmts[1],
                         for_node});

    root_block->as<ir::ScheduleBlockRealize>()
        ->schedule_block->as<ir::ScheduleBlock>()
        ->body = block_node;

    for (auto &tensor : {internal, tmp_out, out}) {
      auto block = ir_sch.GetBlock(tensor->name);
      auto loops = ir_sch.GetLoops(block);
      if (!loops.empty())
        ir_sch.Split(loops[0], {-1, ir::GetLoopExtent(loops[0])});
    }
  }
  auto reshape_block = ir_sch.GetBlock(reshape->name);
  ir_sch.ComputeInline(reshape_block);

  auto internal_block = ir_sch.GetBlock(internal->name);
  ir_sch.SetBuffer(internal_block, "local", true);

  auto tmp_out_block = ir_sch.GetBlock(tmp_out->name);
  ir_sch.SetBuffer(tmp_out_block, "local", true);

  // The current one-dimensional reduce does not make full use of SM.
  // This case is optimized into a two-dimensional.
  auto internal_loops = ir_sch.GetLoops(internal->name);
  auto block_dim_x = internal_loops[1].As<ir::For>()->extent.as_int32();
  int block_dim_y = block_dim_x <= 32 ? 2 : 1;

  for (auto &tensor : {internal, tmp_out, out}) {
    auto loops = ir_sch.GetLoops(tensor->name);
    if (loops.size() == 1) {
      ir_sch.Split(loops[0], {-1, 1});
      loops = ir_sch.GetLoops(tensor->name);
    }
    if (block_dim_y != 1) {
      ir_sch.Split(loops[0], {-1, block_dim_y});
      loops = ir_sch.GetLoops(tensor->name);
      ir_sch.Bind(loops[0], "blockIdx.x");
      ir_sch.Bind(loops[1], "threadIdx.y");
      ir_sch.Bind(loops[2], "threadIdx.x");
    } else {
      ir_sch.Bind(loops[0], "blockIdx.x");
      ir_sch.Bind(loops[1], "threadIdx.x");
    }
  }
  VLOG(3) << "After IRCudaTwoStepReduceSchedule : "
          << ir_sch.GetModule().GetExprs().at(0);
  // ir_sch.SimpleComputeAt(ir_sch.GetBlock(tmp_out->name),
  // ir_sch.GetLoops(out->name)[0]);
  // ir_sch.SimpleComputeAt(ir_sch.GetBlock(internal->name),
  // ir_sch.GetLoops(out->name)[0]);
}

void IRSoftmaxScheduleCPU(ir::IRSchedule &ir_sch, int axis) {  // NOLINT
  ir_sch.MergeExprs();
  auto all_blocks = ir_sch.GetAllBlocks();
  CHECK_EQ(all_blocks.size(), 3U);
  auto output = GetTensor(all_blocks[2]);
  if (axis == -1) {
    axis += output->shape.size();
  }
  auto loops = ir_sch.GetLoops(all_blocks[2]);
  // ir_sch.Parallel(loops[0]);
  all_blocks = ir_sch.GetAllBlocks();
  for (int i = 1; i < axis; ++i) {
    ir_sch.Fuse(all_blocks[2], {0, 1});
  }
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[2]);
  ir_sch.ComputeAt(all_blocks[1], loops[0]);
}

void IRPoolScheduleGPU(ir::IRSchedule &ir_sch,  // NOLINT
                       const common::Target &target,
                       int arg_pack_size) {
  VLOG(3) << "Before IRPoolScheduleGPU: "
          << ir_sch.GetModule().GetExprs().at(0);
  auto all_blocks = ir_sch.GetAllBlocks();
  VLOG(3) << "all_blocks[0] is : " << all_blocks[0];
  auto loops = ir_sch.GetLoops(all_blocks[0]);
  ir_sch.Fuse(loops);
  // Blocks were changed after Fuse, so we have to get all blocks again.
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[0]);
  auto splited = ir_sch.Split(loops[0], {-1, 1024});
  ir_sch.Bind(splited[0], "blockIdx.x");
  ir_sch.Bind(splited[1], "threadIdx.x");
  VLOG(3) << "End IRPoolScheduleGPU: " << ir_sch.GetModule().GetExprs().at(0);
}

void IRGlobalPoolScheduleGPU(ir::IRSchedule &ir_sch,  // NOLINT
                             const common::Target &target) {
  VLOG(3) << "Before IRGlobalPoolScheduleGPU: "
          << ir_sch.GetModule().GetExprs().at(0);
  auto all_blocks = ir_sch.GetAllBlocks();
  CHECK_EQ(all_blocks.size(), 2U);
  auto loops = ir_sch.GetLoops(all_blocks[1]);
  if (loops.size() > 1) {
    auto fused = ir_sch.Fuse(all_blocks[0], {0, 1});
    auto splited = ir_sch.Split(fused, {-1, 32});
    all_blocks = ir_sch.GetAllBlocks();
    fused = ir_sch.Fuse(all_blocks[1], {0, 1});
    splited = ir_sch.Split(fused, {-1, 32});
    ir_sch.Bind(splited[0], "blockIdx.x");
    ir_sch.Bind(splited[1], "threadIdx.y");
    all_blocks = ir_sch.GetAllBlocks();
    ir_sch.SimpleComputeAt(all_blocks[0], splited[1]);
    all_blocks = ir_sch.GetAllBlocks();
    ir_sch.SetBuffer(all_blocks[0], "local", true);
    loops = ir_sch.GetLoops(all_blocks[0]);
    CHECK_GE(loops.size(), 3U);
    ir_sch.Bind(loops[2], "threadIdx.x");
  } else {
    loops = ir_sch.GetLoops(all_blocks[0]);
    auto splited = ir_sch.Split(loops[0], {-1, 32});
    all_blocks = ir_sch.GetAllBlocks();
    loops = ir_sch.GetLoops(all_blocks[1]);
    splited = ir_sch.Split(loops[0], {-1, 32});
    ir_sch.Bind(splited[0], "blockIdx.x");
    ir_sch.Bind(splited[1], "threadIdx.y");
    all_blocks = ir_sch.GetAllBlocks();
    splited = ir_sch.GetLoops(all_blocks[1]);
    ir_sch.SimpleComputeAt(all_blocks[0], splited[1]);
    all_blocks = ir_sch.GetAllBlocks();
    ir_sch.SetBuffer(all_blocks[0], "local", true);
    loops = ir_sch.GetLoops(all_blocks[0]);
    CHECK_GE(loops.size(), 3U);
    ir_sch.Bind(loops[2], "threadIdx.x");
  }
  VLOG(3) << "After IRGlobalPoolScheduleGPU: "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleDepthwiseConv(ir::IRSchedule &ir_sch,  // NOLINT
                                 const std::vector<ir::Expr> &tensors) {
  if (tensors.size() == 3U) {
    CHECK(tensors[1].as_tensor());
    auto input_pad = ir_sch.GetBlock(tensors[1].as_tensor_ref()->name);
    ir_sch.ComputeInline(input_pad);
  }
  auto all_blocks = ir_sch.GetAllBlocks();
  VLOG(3) << "Begin IRCudaScheduleDepthwiseConv with expr: "
          << ir_sch.GetModule().GetExprs().at(0);
  auto OL = ir_sch.CacheWrite(all_blocks[0], 0, "local");
  all_blocks = ir_sch.GetAllBlocks();
  CHECK_GE(all_blocks.size(), 2);
  auto loops = ir_sch.GetLoops(all_blocks[1]);
  CHECK_GE(loops.size(), 4);
  ir_sch.Bind(loops[0], "blockIdx.x");
  ir_sch.Bind(loops[1], "blockIdx.y");
  ir_sch.Bind(loops[2], "blockIdx.z");
  ir_sch.Bind(loops[3], "threadIdx.x");
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[1]);
  ir_sch.ComputeAt(all_blocks[0], loops[3]);
  VLOG(3) << "After IRCudaScheduleDepthwiseConv with expr: "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleConv(ir::IRSchedule &ir_sch,  // NOLINT
                        const common::Target &target) {
  VLOG(3) << "Begin IRCudaScheduleConv with expr: "
          << ir_sch.GetModule().GetExprs().at(0);
  auto &res = ScheduleParam::get_cuda_instance().GetParam();

  auto all_blocks = ir_sch.GetAllBlocks();
  CHECK_EQ(all_blocks.size(), 3U);
  auto input_pad = GetTensor(all_blocks[0]);
  auto output = GetTensor(all_blocks[2]);
  all_blocks = ir_sch.GetAllBlocks();
  CHECK_EQ(all_blocks.size(), 3U);
  auto weights = GetReadTensor(all_blocks[2], 2);

  int n = output->shape[0].as_int32();
  int c = output->shape[1].as_int32();
  optim::Simplify(&(output->shape[2]));
  int h = output->shape[2].as_int32();
  optim::Simplify(&(output->shape[3]));
  int w = output->shape[3].as_int32();
  int rc = input_pad->shape[1].as_int32();

  std::string key = "CudaDirectConvSchedule " +
                    std::to_string(input_pad->shape[0].as_int32()) + " " +
                    std::to_string(input_pad->shape[1].as_int32()) + " " +
                    std::to_string(input_pad->shape[2].as_int32()) + " " +
                    std::to_string(input_pad->shape[3].as_int32()) + " " +
                    std::to_string(weights->shape[0].as_int32()) + " " +
                    std::to_string(weights->shape[1].as_int32()) + " " +
                    std::to_string(weights->shape[2].as_int32()) + " " +
                    std::to_string(weights->shape[3].as_int32()) + " " +
                    std::to_string(output->shape[0].as_int32()) + " " +
                    std::to_string(output->shape[1].as_int32()) + " " +
                    std::to_string(output->shape[2].as_int32()) + " " +
                    std::to_string(output->shape[3].as_int32());
  if (res.count(key) == 0) {
    VLOG(3) << "Didn't find saved param, key is: " << key;
  } else {
    VLOG(3) << "Find saved param! key is: " << key;
    // Todo:temporarily turn off loading params
    // IRCudaScheduleConv2(ir_sch, input_pad, weights, output, target, key);
    // return;
  }
  ir_sch.ComputeInline(all_blocks[0]);
  int f_inner = GetInnerSplitter(c, h);
  int block_z = SplitEven(c / f_inner);
  int thread_z = c / f_inner / block_z;

  int rc_factor = SplitEven(rc);
  while (w * thread_z > 1024 && thread_z % 2 == 0) {
    thread_z = thread_z / 2;
    f_inner = f_inner * 2;
  }
  CHECK_LE(w * thread_z, 1024) << "Wrong Param of Conv2d!";
  std::vector<Expr> loops;
  all_blocks = ir_sch.GetAllBlocks();
  auto reduce_init_name = GetTensor(all_blocks[0])->name;
  {
    // Do CacheWrite
    all_blocks = ir_sch.GetAllBlocks();
    auto OL = ir_sch.CacheWrite(all_blocks[1], 0, "local");
    VLOG(3) << "After CacheWrite with expr: "
            << ir_sch.GetModule().GetExprs().at(0);
  }
  all_blocks = ir_sch.GetAllBlocks();
  auto temp_output_name = GetTensor(all_blocks[1])->name;
  auto final_output_name = GetTensor(all_blocks[2])->name;
  {
    // Do Split
    loops = ir_sch.GetLoops(final_output_name);
    CHECK_GE(loops.size(), 2U);
    ir_sch.Split(loops[1], {-1, thread_z, f_inner});
  }
  {
    // Do Reorder
    loops = ir_sch.GetLoops(final_output_name);
    CHECK_GE(loops.size(), 6U);
    ir_sch.Reorder({loops[1], loops[4], loops[2], loops[5], loops[3]});
  }
  {
    // Do ComputeAt
    auto temp_out = ir_sch.GetBlock(temp_output_name);
    loops = ir_sch.GetLoops(final_output_name);
    CHECK_GE(loops.size(), 5U);
    ir_sch.ComputeAt(temp_out, loops[4]);
  }
  VLOG(3) << "After ComputeAt with expr: "
          << ir_sch.GetModule().GetExprs().at(0);
  {
    // Do Split
    loops = ir_sch.GetLoops(temp_output_name);
    CHECK_GE(loops.size(), 7U);
    ir_sch.Split(loops[6], {-1, rc_factor});
  }
  {
    // Do Split
    auto reduce_init = ir_sch.GetBlock(reduce_init_name);
    ir::ScheduleBlockRealize *reduce_init_block =
        reduce_init.As<ir::ScheduleBlockRealize>();
    loops = ir_sch.GetLoops(reduce_init_name);
    // If loops size is less than 4, it means one or more 1-loops are eliminated
    // in the lowering process. Here we restore them by identifying the constant
    // iter value in the ScheduleBlock
    while (loops.size() < 4U) {
      for (int i = 0; i < reduce_init_block->iter_values.size(); ++i) {
        auto &v = reduce_init_block->iter_values[i];
        if (v.is_constant()) {
          ir_sch.Split(loops[i], {1, -1});
        }
      }
      loops = ir_sch.GetLoops(reduce_init_name);
    }
    CHECK_EQ(loops.size(), 4U);
    ir_sch.Split(loops[1], {-1, thread_z, f_inner});
  }
  {
    // Do Reorder
    loops = ir_sch.GetLoops(reduce_init_name);
    CHECK_GE(loops.size(), 6U);
    ir_sch.Reorder({loops[1], loops[4], loops[2], loops[5], loops[3]});
  }
  VLOG(3) << "After Reorder with expr: " << ir_sch.GetModule().GetExprs().at(0);
  {
    // Do SimpleComputeAt
    auto reduce_init = ir_sch.GetBlock(reduce_init_name);
    loops = ir_sch.GetLoops(temp_output_name);
    CHECK_GE(loops.size(), 6U);
    ir_sch.SimpleComputeAt(reduce_init, loops[5]);
  }
  {
    // Do Bind
    loops = ir_sch.GetLoops(final_output_name);
    CHECK_GE(loops.size(), 5U);
    ir_sch.Bind(loops[1], "blockIdx.z");
    ir_sch.Bind(loops[2], "blockIdx.y");
    ir_sch.Bind(loops[3], "threadIdx.z");
    ir_sch.Bind(loops[4], "threadIdx.x");
  }
  VLOG(3) << "After IRCudaScheduleConv, expr is : "
          << ir_sch.GetModule().GetExprs().at(0);
}

void IRCudaScheduleConv2(ir::IRSchedule &ir_sch,  // NOLINT
                         ir::Tensor &input_pad,   // NOLINT
                         ir::Tensor &weights,     // NOLINT
                         ir::Tensor &output,      // NOLINT
                         const common::Target &target,
                         const std::string &key) {
  auto &res = ScheduleParam::get_cuda_instance().GetParam();

  auto all_blocks = ir_sch.GetAllBlocks();

  // stages[input_pad]->ComputeInline();

  optim::Simplify(&(output->shape[2]));
  optim::Simplify(&(output->shape[3]));

  VLOG(3) << "Begin IRCudaScheduleConv2 with expr : "
          << ir_sch.GetModule().GetExprs().at(0);
  auto input_cache = ir_sch.CacheRead(all_blocks[2], 1, "shared");
  all_blocks = ir_sch.GetAllBlocks();
  auto weights_cache = ir_sch.CacheRead(all_blocks[3], 2, "shared");
  all_blocks = ir_sch.GetAllBlocks();
  auto output_cache = ir_sch.CacheWrite(all_blocks[4], 0, "local");
  all_blocks = ir_sch.GetAllBlocks();
  ir_sch.ComputeInline(all_blocks[1]);
  VLOG(3) << "In the middle of IRCudaScheduleConv2, expr is: "
          << ir_sch.GetModule().GetExprs().at(0);
  auto &x_param = res[key]["x"];
  auto &y_param = res[key]["y"];
  auto &f_param = res[key]["f"];
  auto &rx_param = res[key]["rx"];
  auto &ry_param = res[key]["ry"];
  auto &rc_param = res[key]["rc"];

  all_blocks = ir_sch.GetAllBlocks();
  auto loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 4U);
  ir_sch.Split(loops[3], {-1, x_param[1], x_param[2], x_param[3]});

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 3U);
  ir_sch.Split(loops[2], {-1, y_param[1], y_param[2], y_param[3]});

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 2U);
  ir_sch.Split(loops[1], {-1, f_param[1], f_param[2], f_param[3]});

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 13U);
  ir_sch.Reorder({loops[0],
                  loops[1],
                  loops[5],
                  loops[9],
                  loops[2],
                  loops[6],
                  loops[10],
                  loops[3],
                  loops[7],
                  loops[11],
                  loops[4],
                  loops[8],
                  loops[12]});

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 13U);
  ir_sch.Bind(loops[1], "blockIdx.z");
  ir_sch.Bind(loops[2], "blockIdx.y");
  ir_sch.Bind(loops[3], "blockIdx.x");
  ir_sch.Bind(loops[7], "threadIdx.z");
  ir_sch.Bind(loops[8], "threadIdx.y");
  ir_sch.Bind(loops[9], "threadIdx.x");
  ir_sch.Unroll(loops[10]);
  ir_sch.Unroll(loops[11]);
  ir_sch.Unroll(loops[12]);

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[4]);
  CHECK_GE(loops.size(), 10U);
  ir_sch.ComputeAt(all_blocks[3], loops[9]);

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 16U);
  ir_sch.Split(loops[15], {-1, rx_param[1]});
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 15U);
  ir_sch.Split(loops[14], {-1, ry_param[1]});
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 14U);
  ir_sch.Split(loops[13], {-1, rc_param[1]});
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 14U);
  ir_sch.Reorder({loops[13],
                  loops[15],
                  loops[17],
                  loops[14],
                  loops[16],
                  loops[18],
                  loops[10],
                  loops[11],
                  loops[12]});

  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 13U);
  ir_sch.ComputeAt(all_blocks[0], loops[12]);
  all_blocks = ir_sch.GetAllBlocks();
  loops = ir_sch.GetLoops(all_blocks[3]);
  CHECK_GE(loops.size(), 13U);
  ir_sch.ComputeAt(all_blocks[1], loops[12]);
  // Work In Progress
  VLOG(3) << "After IRCudaScheduleConv2, expr is: "
          << ir_sch.GetModule().GetExprs().at(0);
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
