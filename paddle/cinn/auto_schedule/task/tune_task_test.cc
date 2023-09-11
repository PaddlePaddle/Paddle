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

#include "paddle/cinn/auto_schedule/task/tune_task.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>

#include "paddle/cinn/auto_schedule/task/task_creator.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/target.h"
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/graph.h"
#include "paddle/cinn/hlir/framework/node.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/hlir/framework/pass.h"
#include "paddle/cinn/hlir/framework/scope.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_printer.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace auto_schedule {

using ::cinn::frontend::NetBuilder;
using ::cinn::frontend::Program;
using ::cinn::hlir::framework::OpLowerer;

Program CreateAddProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);
  auto program = builder.Build();

  return program;
}

TEST(TuneTask, GraphToUnoptLoweredFunc_NoPass) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  Program prog = CreateAddProgram();
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);

  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph.get());
  ASSERT_EQ(tasks.size(), 2UL);

  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);

  std::stringstream ss;
  for (TuneTask& task : tasks) {
    task.Initialize(shape_dict, dtype_dict, &op_lowerer);

    std::vector<ir::Expr> exprs = task.GetLoweredFuncBodyExprs();
    VLOG(6) << "ir:Expr is: ";
    for (const ir::Expr& e : exprs) {
      VLOG(6) << e;
      ss << e << std::endl;
    }
  }

  std::string expr_str = ss.str();
#ifdef CINN_WITH_CUDA
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 24)
      {
        ScheduleBlock(var_1)
        {
          i0, i1 = axis.bind(i, j)
          var_1[i, j] = (A[i, j] + B[i, j])
        }
      }
    }
  }
}
{
  ScheduleBlock(root_0)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 24)
      {
        ScheduleBlock(var_2)
        {
          i0_0, i1_0 = axis.bind(i, j)
          var_2[i, j] = (A[i, j] + var_1[i, j])
        }
      }
    }
  }
}
)ROC";
#else
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 24)
      {
        ScheduleBlock(var_1)
        {
          i0, i1 = axis.bind(i, j)
          var_1[i0, i1] = (A[i0, i1] + B[i0, i1])
        }
      }
    }
  }
}
{
  ScheduleBlock(root_0)
  {
    serial for (i, 0, 32)
    {
      serial for (j, 0, 24)
      {
        ScheduleBlock(var_2)
        {
          i0_0, i1_0 = axis.bind(i, j)
          var_2[i0_0, i1_0] = (A[i0_0, i1_0] + var_1[i0_0, i1_0])
        }
      }
    }
  }
}
)ROC";
#endif

  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

TEST(TuneTask, GraphToUnoptLoweredFunc_ApplyPass) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  Program prog = CreateAddProgram();
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);
  ApplyPass(graph.get(), "OpFusionPass");

  TaskCreator task_creator;
  std::vector<TuneTask> tasks = task_creator.CreateTuneTaskOpLevel(graph.get());

  ASSERT_EQ(tasks.size(), 1UL);

  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");

  OpLowerer op_lowerer(
      new hlir::framework::OpLowererImpl(dtype_dict, shape_dict, target));

  std::stringstream ss;
  for (TuneTask& task : tasks) {
    task.Initialize(shape_dict, dtype_dict, &op_lowerer);

    std::vector<ir::Expr> exprs = task.GetLoweredFuncBodyExprs();
    VLOG(6) << "ir:Expr is: ";
    for (const ir::Expr& e : exprs) {
      VLOG(6) << e;
      ss << e << std::endl;
    }
  }

  std::string expr_str = ss.str();
#ifdef CINN_WITH_CUDA
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 24)
        {
          ScheduleBlock(var_1)
          {
            i0, i1 = axis.bind(i, j)
            var_1[i, j] = (A[i, j] + B[i, j])
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 24)
        {
          ScheduleBlock(var_2)
          {
            i0_0, i1_0 = axis.bind(i, j)
            var_2[i, j] = (A[i, j] + var_1[i, j])
          }
        }
      }
    }
  }
}
)ROC";

#else
  std::string target_str = R"ROC(
{
  ScheduleBlock(root)
  {
    {
      serial for (i, 0, 32)
      {
        serial for (j, 0, 24)
        {
          ScheduleBlock(var_1)
          {
            i0, i1 = axis.bind(i, j)
            var_1[i0, i1] = (A[i0, i1] + B[i0, i1])
          }
        }
      }
      serial for (i, 0, 32)
      {
        serial for (j, 0, 24)
        {
          ScheduleBlock(var_2)
          {
            i0_0, i1_0 = axis.bind(i, j)
            var_2[i0_0, i1_0] = (A[i0_0, i1_0] + var_1[i0_0, i1_0])
          }
        }
      }
    }
  }
}
)ROC";
#endif

  EXPECT_EQ(utils::Trim(target_str), utils::Trim(expr_str));
}

TEST(TuneTask, SerializeToString) {
  Context::Global().ResetNameId();
#ifdef CINN_WITH_CUDA
  Target target = common::DefaultNVGPUTarget();
#else
  Target target = common::DefaultHostTarget();
#endif
  Program prog = CreateAddProgram();
  auto graph = std::make_shared<hlir::framework::Graph>(prog, target);

  TaskCreator task_creator;
  std::vector<TuneTask> single_tasks =
      task_creator.CreateTuneTaskOpLevel(graph.get());

  const auto& shape_dict = graph->GetAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  const auto& dtype_dict =
      graph->GetAttrs<absl::flat_hash_map<std::string, common::Type>>(
          "inferdtype");
  OpLowerer op_lowerer(
      new hlir::framework::OpLowererImpl(dtype_dict, shape_dict, target));
  ASSERT_EQ(single_tasks.size(), 2UL);
  for (auto&& task : single_tasks) {
    task.Initialize(shape_dict, dtype_dict, &op_lowerer);
  }

#ifdef CINN_WITH_CUDA
  std::string single_add_str = R"ROC(Target<linux,nvgpu,64>

Group {
  (var_1->float32[32,24]) = elementwise_add(A->float32[32,24], B->float32[32,24])
}
)ROC";
#else
  std::string single_add_str = R"ROC(Target<linux,x86,64>

Group {
  (var_1->float32[32,24]) = elementwise_add(A->float32[32,24], B->float32[32,24])
}
)ROC";
#endif
  EXPECT_EQ(single_tasks[0].serialized_key, single_add_str);

  ApplyPass(graph.get(), "OpFusionPass");
  std::vector<TuneTask> fused_tasks =
      task_creator.CreateTuneTaskOpLevel(graph.get());
  ASSERT_EQ(fused_tasks.size(), 1UL);
  fused_tasks[0].Initialize(shape_dict, dtype_dict, &op_lowerer);

#ifdef CINN_WITH_CUDA
  std::string fused_expected_str = R"ROC(Target<linux,nvgpu,64>

Group {
  (var_1->float32[32,24]) = elementwise_add(A->float32[32,24], B->float32[32,24])
  (var_2->float32[32,24]) = elementwise_add(A->float32[32,24], var_1->float32[32,24])
}
)ROC";
#else
  std::string fused_expected_str = R"ROC(Target<linux,x86,64>

Group {
  (var_1->float32[32,24]) = elementwise_add(A->float32[32,24], B->float32[32,24])
  (var_2->float32[32,24]) = elementwise_add(A->float32[32,24], var_1->float32[32,24])
}
)ROC";
#endif
  EXPECT_EQ(fused_tasks[0].serialized_key, fused_expected_str);
}

}  // namespace auto_schedule
}  // namespace cinn
