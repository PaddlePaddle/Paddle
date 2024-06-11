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

#include "paddle/cinn/ir/schedule_block_graph.h"
#include <gtest/gtest.h>
#include "paddle/cinn/frontend/net_builder.h"
#include "paddle/cinn/frontend/optimize.h"
#include "paddle/cinn/frontend/syntax.h"
#include "paddle/cinn/hlir/framework/op_lowering.h"
#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/common/enforce.h"
PD_DECLARE_bool(cinn_new_group_scheduler);

namespace cinn {
namespace ir {

IRSchedule MakeIRSchedule(frontend::Program* program) {
#ifdef CINN_WITH_CUDA
  Target target = cinn::common::DefaultNVGPUTarget();
#else
  Target target = cinn::common::DefaultHostTarget();
#endif
  std::unordered_set<std::string> fetch_ids;
  auto graph = frontend::Optimize(program, fetch_ids, target);
  LOG_IF(WARNING, graph->fusion_groups.size() > 1)
      << "Test Graph has more than 1 group";
  auto& dtype_dict = graph->GetMutableAttrs<
      absl::flat_hash_map<std::string, cinn::common::Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<
      absl::flat_hash_map<std::string, hlir::framework::shape_t>>("infershape");
  auto op_lowerer =
      hlir::framework::CreateOpLowerer(dtype_dict, shape_dict, target);

  std::vector<LoweredFunc> lowered_funcs =
      op_lowerer.Lower(graph->fusion_groups.front(), false, false);
  CHECK(!lowered_funcs.empty()) << "lowered_funcs_ is empty";

  std::vector<Expr> bodys;
  for (auto&& func : lowered_funcs) {
    bodys.emplace_back(func->body);
  }
  return IRSchedule(ModuleExpr({std::move(bodys)}), 1);
}

std::string GetIR(const ir::IRSchedule& schedule) {
  const auto& exprs = schedule.GetModule().GetExprs();
  std::stringstream module_stream;
  for (auto i = 0; i < exprs.size(); ++i) {
    module_stream << "Expr " << i << " {\n"
                  << exprs.at(i) << "\n}  // end Expr " << i << "\n";
  }
  return module_stream.str();
}

frontend::Program CreateElementwiseProgram() {
  constexpr int M = 32;
  constexpr int N = 24;

  frontend::NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.Add(a, c);
  auto e = builder.Relu(c);
  auto f = builder.Relu(d);
  auto program = builder.Build();

  return program;
}

frontend::Program CreateReduceProgram() {
  constexpr int M = 64;
  constexpr int N = 128;

  frontend::NetBuilder builder("net_builder");
  auto a = builder.CreateInput(Float(32), {M, N}, "A");
  auto b = builder.CreateInput(Float(32), {M, N}, "B");
  auto c = builder.Add(a, b);
  auto d = builder.ReduceSum(c, {0});
  auto e = builder.BroadcastTo(d, {M, N});
  auto f = builder.Add(e, a);
  auto program = builder.Build();

  return program;
}

TEST(ScheduleBlockGraph, elementwise) {
  Context::Global().ResetNameId();
  frontend::Program program = CreateElementwiseProgram();
  IRSchedule ir_sch = MakeIRSchedule(&program);
  LOG(INFO) << GetIR(ir_sch);
  ScheduleBlockGraph sbg(ir_sch);
  LOG(INFO) << sbg.Visualize();
  PADDLE_ENFORCE_EQ(
      sbg.BlockIdsInOrder().size(),
      6,
      phi::errors::InvalidArgument("The size of BlockIdsInOrder should be 6!"));
  PADDLE_ENFORCE_EQ(
      sbg.nodes().size(),
      6,
      phi::errors::InvalidArgument("The size of nodes should be 6!"));

  ScheduleBlockNode* v2 = sbg.RetrieveNode("var_2");
  CHECK(v2);
  PADDLE_ENFORCE_EQ(
      v2->UpstreamNodes().size(),
      1,
      phi::errors::InvalidArgument("The size of UpstreamNodes should be 1!"));
  PADDLE_ENFORCE_EQ(
      v2->DownstreamNodes().size(),
      1,
      phi::errors::InvalidArgument("The size of DownstreamNodes should be 1!"));

  ScheduleBlockNode* v4 = sbg.RetrieveNode("var_4");
  CHECK(v4);
  PADDLE_ENFORCE_EQ(
      v4->UpstreamNodes().size(),
      3,
      phi::errors::InvalidArgument("The size of UpstreamNodes should be 3!"));
  PADDLE_ENFORCE_EQ(
      v4->DownstreamNodes().size(),
      0,
      phi::errors::InvalidArgument("The size of DownstreamNodes should be 0!"));

  std::vector<std::string> reverse_dfs_topo_order_ids;
  sbg.DFSTopoWalk([&reverse_dfs_topo_order_ids](const ScheduleBlockNode* node) {
    reverse_dfs_topo_order_ids.push_back(node->id());
  });
  for (const std::string& id : reverse_dfs_topo_order_ids) {
    LOG(INFO) << id;
  }
  PADDLE_ENFORCE_EQ(reverse_dfs_topo_order_ids.size(),
                    6,
                    phi::errors::InvalidArgument(
                        "The size of reverse_dfs_topo_order_ids should be 6!"));

  std::vector<std::string> dfs_topo_order_ids;
  sbg.DFSTopoWalk(
      [&dfs_topo_order_ids](const ScheduleBlockNode* node) {
        dfs_topo_order_ids.push_back(node->id());
      },
      false);
  for (const std::string& id : dfs_topo_order_ids) {
    LOG(INFO) << id;
  }
  PADDLE_ENFORCE_EQ(dfs_topo_order_ids.size(),
                    6,
                    phi::errors::InvalidArgument(
                        "The size of dfs_topo_order_ids should be 6!"));
}

#ifdef CINN_WITH_CUDA
TEST(ScheduleBlockGraph, reduce) {
  if (FLAGS_cinn_new_group_scheduler) {
    Context::Global().ResetNameId();
    frontend::Program program = CreateReduceProgram();
    IRSchedule ir_sch = MakeIRSchedule(&program);
    ScheduleBlockGraph sbg(ir_sch);
    LOG(INFO) << GetIR(ir_sch);
    LOG(INFO) << sbg.Visualize();
    PADDLE_ENFORCE_EQ(sbg.BlockIdsInOrder().size(),
                      5,
                      phi::errors::InvalidArgument(
                          "The size of BlockIdsInOrder should be 5!"));
    PADDLE_ENFORCE_EQ(
        sbg.nodes().size(),
        5,
        phi::errors::InvalidArgument("The size of nodes should be 5!"));

    ScheduleBlockNode* v_reduce_init = sbg.RetrieveNode("var_2__reduce_init");
    CHECK(v_reduce_init);
    PADDLE_ENFORCE_EQ(
        v_reduce_init->UpstreamNodes().size(),
        0,
        phi::errors::InvalidArgument("The size of UpstreamNodes should be 0!"));
    PADDLE_ENFORCE_EQ(v_reduce_init->DownstreamNodes().size(),
                      3,
                      phi::errors::InvalidArgument(
                          "The size of DownstreamNodes should be 3!"));

    ScheduleBlockNode* v = sbg.RetrieveNode("var_2");
    CHECK(v);
    PADDLE_ENFORCE_EQ(
        v->UpstreamNodes().size(),
        2,
        phi::errors::InvalidArgument("The size of UpstreamNodes should be 2!"));
    PADDLE_ENFORCE_EQ(v->DownstreamNodes().size(),
                      2,
                      phi::errors::InvalidArgument(
                          "The size of DownstreamNodes should be 2!"));

    std::vector<std::string> reverse_dfs_topo_order_ids;
    sbg.DFSTopoWalk(
        [&reverse_dfs_topo_order_ids](const ScheduleBlockNode* node) {
          reverse_dfs_topo_order_ids.push_back(node->id());
        });
    for (const std::string& id : reverse_dfs_topo_order_ids) {
      LOG(INFO) << id;
    }
    PADDLE_ENFORCE_EQ(
        reverse_dfs_topo_order_ids.size(),
        5,
        phi::errors::InvalidArgument(
            "The size of reverse_dfs_topo_order_ids should be 5!"));

    std::vector<std::string> dfs_topo_order_ids;
    sbg.DFSTopoWalk(
        [&dfs_topo_order_ids](const ScheduleBlockNode* node) {
          dfs_topo_order_ids.push_back(node->id());
        },
        false);
    for (const std::string& id : dfs_topo_order_ids) {
      LOG(INFO) << id;
    }
    PADDLE_ENFORCE_EQ(dfs_topo_order_ids.size(),
                      5,
                      phi::errors::InvalidArgument(
                          "The size of dfs_topo_order_ids should be 5!"));
  }
}

TEST(ScheduleBlockGraph, arg_max) {
  Context::Global().ResetNameId();
  frontend::NetBuilder builder("net_builder");
  auto x = builder.CreateInput(Float(32), {8, 16}, "X");
  auto y = builder.Argmax(x, 0);
  frontend::Program program = builder.Build();

  IRSchedule ir_sch = MakeIRSchedule(&program);
  LOG(INFO) << GetIR(ir_sch);
  ScheduleBlockGraph sbg(ir_sch);
  LOG(INFO) << sbg.Visualize();
  PADDLE_ENFORCE_EQ(
      sbg.BlockIdsInOrder().size(),
      3,
      phi::errors::InvalidArgument("The size of BlockIdsInOrder should be 3!"));
  PADDLE_ENFORCE_EQ(
      sbg.nodes().size(),
      3,
      phi::errors::InvalidArgument("The size of nodes should be 3!"));

  ScheduleBlockNode* v0_idx = sbg.RetrieveNode("var_0_index");
  CHECK(v0_idx);
  PADDLE_ENFORCE_EQ(
      v0_idx->UpstreamNodes().size(),
      1,
      phi::errors::InvalidArgument("The size of UpstreamNodes should be 1!"));
  PADDLE_ENFORCE_EQ(
      v0_idx->DownstreamNodes().size(),
      1,
      phi::errors::InvalidArgument("The size of DownstreamNodes should be 1!"));

  ScheduleBlockNode* v0 = sbg.RetrieveNode("var_0");
  CHECK(v0);
  PADDLE_ENFORCE_EQ(
      v0->UpstreamNodes().size(),
      2,
      phi::errors::InvalidArgument("The size of UpstreamNodes should be 2!"));
  PADDLE_ENFORCE_EQ(
      v0->DownstreamNodes().size(),
      0,
      phi::errors::InvalidArgument("The size of DownstreamNodes should be 0!"));

  std::vector<std::string> reverse_dfs_topo_order_ids;
  sbg.DFSTopoWalk([&reverse_dfs_topo_order_ids](const ScheduleBlockNode* node) {
    reverse_dfs_topo_order_ids.push_back(node->id());
  });
  for (const std::string& id : reverse_dfs_topo_order_ids) {
    LOG(INFO) << id;
  }
  PADDLE_ENFORCE_EQ(reverse_dfs_topo_order_ids.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The size of reverse_dfs_topo_order_ids should be 3!"));

  std::vector<std::string> dfs_topo_order_ids;
  sbg.DFSTopoWalk(
      [&dfs_topo_order_ids](const ScheduleBlockNode* node) {
        dfs_topo_order_ids.push_back(node->id());
      },
      false);
  for (const std::string& id : dfs_topo_order_ids) {
    LOG(INFO) << id;
  }
  PADDLE_ENFORCE_EQ(dfs_topo_order_ids.size(),
                    3,
                    phi::errors::InvalidArgument(
                        "The size of dfs_topo_order_ids should be 3!"));
}
#endif

}  // namespace ir
}  // namespace cinn
