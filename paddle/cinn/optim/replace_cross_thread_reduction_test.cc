// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/replace_cross_thread_reduction.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/op/ir_operators.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace optim {

TEST(CrossThreadReductionReplacer, basic) {
#ifdef CINN_WITH_CUDA
  Context::Global().ResetNameId();
  Placeholder<float> A("A", {Expr(64), Expr(128)});
  Target target = common::DefaultNVGPUTarget();
  Module::Builder builder("reduce_sum", target);
  Var reduce_j(128, "reduce_j");
  ir::Tensor B = Compute(
      {Expr(64)},
      [&](Var i) { return lang::ReduceSum(A(i, reduce_j), {reduce_j}); },
      "B");
  ast_gen_ius::TensorGroup tensor_group({A, B});
  auto func = lang::LowerToAst("reduce_sum", {A, B}, &tensor_group);
  VLOG(6) << "original func\n" << func;

  ir::ModuleExpr mod_expr({func->body});
  ir::IRSchedule ir_sch(mod_expr);

  ir_sch.Bind(ir_sch.GetLoops("B")[0], "blockIdx.x");
  ir_sch.Bind(ir_sch.GetLoops("B")[1], "threadIdx.x");

  ir::Expr new_func = ir_sch.GetModule().GetExprs()[0];
  VLOG(6) << "After Bind: " << new_func;

  ReplaceCrossThreadReduction(&new_func);
  VLOG(6) << "After ReplaceCrossThreadReduction: " << new_func;

  EXPECT_EQ(utils::GetStreamCnt(new_func), utils::Trim(R"ROC({
  ScheduleBlock(root)
  {
    thread_bind[blockIdx.x] for (i, 0, 64)
    {
      ScheduleBlock(B__reduce_init)
      {
        i0 = axis.bind(i)
        B__reduce_init[i0] = 0.00000000f
      }
      thread_bind[threadIdx.x] for (reduce_j, 0, 128)
      {
        ScheduleBlock(B)
        {
          i0_0, i1 = axis.bind(i, reduce_j)
          B[i0_0] = cinn_block_reduce_sum_fp32_internal(A[i0_0, i1])
        }
      }
    }
  }
}
)ROC"));
#endif
}

}  // namespace optim
}  // namespace cinn
