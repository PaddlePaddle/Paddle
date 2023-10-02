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

#include "paddle/cinn/auto_schedule/post_schedule_rule/cooperative_process.h"

#include <gtest/gtest.h>

#include "paddle/cinn/auto_schedule/search_space/auto_gen_rule/test_helper.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "test/cpp/cinn/program_builder.h"

namespace cinn {
namespace auto_schedule {

class TestCooperativeProcess : public TestAutoGenRuleBase {
 public:
  int fixed_rand_seed = 1;
  std::vector<std::string> default_input_names;
  std::vector<std::string> default_output_names;
};

TEST_F(TestCooperativeProcess, Matmul) {
  default_input_names = {"X", "Y"};
  default_output_names = {"temp_matmul_out"};
  std::vector<int32_t> X_shape = {32, 32};
  std::vector<int32_t> Y_shape = {32, 32};
  std::vector<int32_t> out_shape = {32, 32};

  int num_blocks_y = 2;
  int num_blocks_x = 2;
  int num_threads_y = 8;
  int num_threads_x = 2;
  int steps_k = 8;

  Initialize(common::DefaultNVGPUTarget());
  frontend::Program matmul_op =
      tests::OpBuilder("matmul").Build({{"X", X_shape}, {"Y", Y_shape}});
  ir::IRSchedule ir_schedule = MakeIRSchedule(matmul_op, fixed_rand_seed);

  // split loops
  std::vector<ir::Expr> loops = ir_schedule.GetLoops("temp_matmul_out");
  std::vector<ir::Expr> k_loops = ir_schedule.Split(loops[2], {steps_k, -1});
  std::vector<ir::Expr> j_loops =
      ir_schedule.Split(loops[1], {num_blocks_x, num_threads_x, -1});
  std::vector<ir::Expr> i_loops =
      ir_schedule.Split(loops[0], {num_blocks_y, num_threads_y, -1});
  // reorder to "SSRRS": i0, j0, i1, j1, k0, k1, j2, i2
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.Reorder({loops[0],
                       loops[3],
                       loops[1],
                       loops[4],
                       loops[6],
                       loops[7],
                       loops[2],
                       loops[5]});
  // fuse and bind
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir::Expr i1_j1_fused = ir_schedule.Fuse({loops[2], loops[3]});
  ir::Expr i0_j0_fused = ir_schedule.Fuse({loops[0], loops[1]});
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.Bind(loops[1], "threadIdx.x");
  ir_schedule.Bind(loops[0], "blockIdx.x");
  // cache read
  ir::Expr out_block = ir_schedule.GetBlock("temp_matmul_out");
  ir::Expr X_cache_block = ir_schedule.CacheRead(out_block, 1, "shared");
  std::string X_cache_block_name = X_cache_block.As<ir::ScheduleBlockRealize>()
                                       ->schedule_block.As<ir::ScheduleBlock>()
                                       ->name;
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.ComputeAt(X_cache_block, loops[2]);
  std::vector<ir::Expr> X_cache_loops =
      ir_schedule.GetLoops(X_cache_block_name);
  ir_schedule.Fuse({X_cache_loops[3], X_cache_loops[4]});
  ir_schedule.Annotate(ir_schedule.GetBlock(X_cache_block_name),
                       ir::attr::cooperative_process,
                       0);

  out_block = ir_schedule.GetBlock("temp_matmul_out");
  ir::Expr Y_cache_block = ir_schedule.CacheRead(out_block, 2, "shared");
  std::string Y_cache_block_name = Y_cache_block.As<ir::ScheduleBlockRealize>()
                                       ->schedule_block.As<ir::ScheduleBlock>()
                                       ->name;
  loops = ir_schedule.GetLoops("temp_matmul_out");
  ir_schedule.ComputeAt(Y_cache_block, loops[2]);
  std::vector<ir::Expr> Y_cache_loops =
      ir_schedule.GetLoops(Y_cache_block_name);
  ir_schedule.Fuse({Y_cache_loops[3], Y_cache_loops[4]});
  ir_schedule.Annotate(ir_schedule.GetBlock(Y_cache_block_name),
                       ir::attr::cooperative_process,
                       0);

  // apply CooperativeProcess
  CooperativeProcess cooperative_process;
  cooperative_process.Apply(&ir_schedule);

  // check ir
  auto ir = GetIR(ir_schedule);
  VLOG(6) << "after CooperativeProcess, ir: \n" << ir;
  std::string expected_ir = R"ROC(Expr 0 {
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 2)
      {
        serial for (j, 0, 2)
        {
          serial for (i_0, 0, 8)
          {
            serial for (j_0, 0, 2)
            {
              serial for (i_1, 0, 2)
              {
                serial for (j_1, 0, 8)
                {
                  ScheduleBlock(temp_matmul_out__reduce_init)
                  {
                    i0, i1 = axis.bind(((16 * i) + ((2 * i_0) + i_1)), ((16 * j) + ((8 * j_0) + j_1)))
                    {
                      temp_matmul_out__reduce_init[((16 * i) + ((2 * i_0) + i_1)), ((16 * j) + ((8 * j_0) + j_1))] = 0.00000000f
                    }
                  }
                }
              }
            }
          }
        }
      }
      thread_bind[blockIdx.x] for (i_j_fused, 0, 4)
      {
        thread_bind[threadIdx.x] for (i_0_j_0_fused, 0, 16)
        {
          serial for (reduce_k_0, 0, 8)
          {
            serial for (ax0_0_ax1_0_fused, 0, 2)
            {
              thread_bind[threadIdx.x] for (ax0_0_ax1_0_fused_0, 0, 16)
              {
                ScheduleBlock(Y_reshape_shared_temp_buffer)
                {
                  v0, v1 = axis.bind(((((16 * ax0_0_ax1_0_fused) + ax0_0_ax1_0_fused_0) / 8) + (4 * reduce_k_0)), ((((16 * ax0_0_ax1_0_fused) + ax0_0_ax1_0_fused_0) % 8) + ((8 * (i_0_j_0_fused % 2)) + (16 * (i_j_fused % 2)))))
                  attrs(compute_at_extra_var:ax0_0,ax1_0)
                  {
                    Y_reshape_shared_temp_buffer[v0, v1] = Y_reshape[v0, v1]
                  }
                }
              }
            }
            __syncthreads()
            thread_bind[threadIdx.x] for (ax0_ax1_fused, 0, 8)
            {
              ScheduleBlock(X_reshape_shared_temp_buffer)
              {
                v0, v1 = axis.bind(((ax0_ax1_fused / 4) + ((2 * (i_0_j_0_fused / 2)) + (16 * (i_j_fused / 2)))), ((ax0_ax1_fused % 4) + (4 * reduce_k_0)))
                attrs(compute_at_extra_var:ax0,ax1)
                {
                  X_reshape_shared_temp_buffer[v0, v1] = X_reshape[v0, v1]
                }
              }
            }
            __syncthreads()
            serial for (reduce_k_1, 0, 4)
            {
              serial for (i_1, 0, 2)
              {
                serial for (j_1, 0, 8)
                {
                  ScheduleBlock(temp_matmul_out)
                  {
                    i0_0, i1_0, i2 = axis.bind(((2 * (i_0_j_0_fused / 2)) + ((16 * (i_j_fused / 2)) + i_1)), ((8 * (i_0_j_0_fused % 2)) + ((16 * (i_j_fused % 2)) + j_1)), ((4 * reduce_k_0) + reduce_k_1))
                    {
                      temp_matmul_out[((2 * (i_0_j_0_fused / 2)) + ((16 * (i_j_fused / 2)) + i_1)), ((8 * (i_0_j_0_fused % 2)) + ((16 * (i_j_fused % 2)) + j_1))] = (temp_matmul_out[((2 * (i_0_j_0_fused / 2)) + ((16 * (i_j_fused / 2)) + i_1)), ((8 * (i_0_j_0_fused % 2)) + ((16 * (i_j_fused % 2)) + j_1))] + (X_reshape_shared_temp_buffer[((2 * (i_0_j_0_fused / 2)) + ((16 * (i_j_fused / 2)) + i_1)), ((4 * reduce_k_0) + reduce_k_1)] * Y_reshape_shared_temp_buffer[((4 * reduce_k_0) + reduce_k_1), ((8 * (i_0_j_0_fused % 2)) + ((16 * (i_j_fused % 2)) + j_1))]))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
}  // end Expr 0
)ROC";
  ASSERT_EQ(ir, expected_ir);

  // build ir::Module and debug source code
  auto ir_module = BuildIRModule(ir_schedule);
  auto source_code = GenSourceCode(ir_module);
  VLOG(6) << "scheduled source code:\n" << source_code;

  // execute and check precision
  CheckResult(
      GenExecutableKernel(ir_module),
      GenExecutableKernel(BuildIRModule(MakeIRSchedule(
          matmul_op, fixed_rand_seed, /* apply_manual_schedule*/ true))),
      default_input_names,
      default_output_names,
      {X_shape, Y_shape},
      {out_shape},
      target_);
}

}  // namespace auto_schedule
}  // namespace cinn
