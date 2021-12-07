//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest.h"
#include "gtest/gtest_pred_impl.h"

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type_inference.h"
#include "paddle/fluid/platform/place.h"

DECLARE_bool(convert_all_blocks);

namespace paddle {
namespace framework {

class SumOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "").AsDuplicable();
    AddOutput("Out", "").AsDuplicable();
    AddComment("");
  }
};

class SumOpWithKernel : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {}
  OpKernelType GetExpectedKernelType(
      const ExecutionContext &ctx) const override {
    return OpKernelType(proto::VarType::FP32, ctx.Input<Tensor>("X")->place());
  }
};

}  // namespace framework
}  // namespace paddle

REGISTER_OP_WITHOUT_GRADIENT(sum, paddle::framework::SumOpWithKernel,
                             paddle::framework::SumOpMaker);

namespace paddle {
namespace framework {
namespace details {

static std::vector<platform::Place> CreatePlaces(size_t num, bool use_cuda) {
  std::vector<platform::Place> result;
  result.reserve(num);
  for (size_t i = 0; i < num; ++i) {
    if (use_cuda) {
      result.emplace_back(platform::CUDAPlace(i));
    } else {
      result.emplace_back(platform::CPUPlace());
    }
  }
  return result;
}

void BuildStrategyApply(BuildStrategy *build_strategy, ir::Graph *graph) {
  std::string loss_name = "";
  Scope scope;
  std::vector<Scope *> scopes = {&scope};

  auto places = CreatePlaces(1, false);
  auto device = platform::Place2DeviceType(places[0]);

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  platform::NCCLCommunicator ctxs;
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
  platform::BKCLCommunicator ctxs;
#endif

  build_strategy->Apply(graph, places, loss_name, scopes, 1,
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
                        device, &ctxs);
#elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_BKCL)
                        device, &ctxs);
#else
                        device);
#endif
}

std::unique_ptr<ir::Graph> CreateGraph() {
  ProgramDesc prog;
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a1"});
  op->SetOutput("Out", {"b1"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("a1")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b1")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  return g;
}

std::unique_ptr<ir::Graph> CreateMultiGraph() {
  ProgramDesc prog;
  prog.AppendBlock(prog.Block(0));
  prog.AppendBlock(prog.Block(0));

  // Set contents in block_0.
  auto *op = prog.MutableBlock(0)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"test_a", "test_b", "test_c"});
  op->SetOutput("Out", {"test_out"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(0)->Var("test_a")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_c")->SetType(proto::VarType::SELECTED_ROWS);
  prog.MutableBlock(0)->Var("test_out");
  op->InferVarType(prog.MutableBlock(0));

  prog.MutableBlock(0)->Var("test_b")->SetType(proto::VarType::LOD_TENSOR);
  op->InferVarType(prog.MutableBlock(0));

  // Set contents in block_1.
  op = prog.MutableBlock(1)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a1"});
  op->SetOutput("Out", {"b1"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(1)->Var("a1")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(1)->Var("b1")->SetType(proto::VarType::LOD_TENSOR);

  // Set contents in block_2.
  op = prog.MutableBlock(2)->AppendOp();
  op->SetType("sum");
  op->SetInput("X", {"a2"});
  op->SetOutput("Out", {"b2"});
  op->SetAttr("op_role", 1);

  prog.MutableBlock(2)->Var("a2")->SetType(proto::VarType::LOD_TENSOR);
  prog.MutableBlock(2)->Var("b2")->SetType(proto::VarType::LOD_TENSOR);

  std::unique_ptr<ir::Graph> g(new ir::Graph(prog));
  return g;
}

inline bool CheckSubGraphSame(ir::Graph *g1, ir::Graph *g2) {
  const auto &g1_nodes_set = g1->Nodes();
  const auto &g2_nodes_set = g2->Nodes();

  if (g1_nodes_set.size() != g2_nodes_set.size()) return false;

  std::vector<ir::Node *> g1_nodes(g1_nodes_set.begin(), g1_nodes_set.end());
  std::vector<ir::Node *> g2_nodes(g2_nodes_set.begin(), g2_nodes_set.end());

  auto comp = [](ir::Node *n1, ir::Node *n2) {
    return n1->Name() > n2->Name();
  };
  std::stable_sort(g1_nodes.begin(), g1_nodes.end(), comp);
  std::stable_sort(g2_nodes.begin(), g2_nodes.end(), comp);

  for (size_t i = 0; i < g1_nodes.size(); ++i) {
    const auto &n1 = g1_nodes[i];
    const auto &n2 = g2_nodes[i];

    if (n1->NodeType() != n2->NodeType()) return false;
    if (n1->Name() != n2->Name()) return false;

    auto n1_inputs = n1->inputs;
    auto n2_inputs = n2->inputs;
    if (n1_inputs.size() != n2_inputs.size()) return false;

    std::stable_sort(n1_inputs.begin(), n1_inputs.end(), comp);
    std::stable_sort(n2_inputs.begin(), n2_inputs.end(), comp);
    for (size_t i = 0; i < n1_inputs.size(); ++i) {
      if (n1_inputs[i]->Name() != n2_inputs[i]->Name()) return false;
    }

    auto n1_outputs = n1->outputs;
    auto n2_outputs = n2->outputs;
    if (n1_outputs.size() != n2_outputs.size()) return false;

    std::stable_sort(n1_outputs.begin(), n1_outputs.end(), comp);
    std::stable_sort(n2_outputs.begin(), n2_outputs.end(), comp);
    for (size_t i = 0; i < n1_outputs.size(); ++i) {
      if (n1_outputs[i]->Name() != n2_outputs[i]->Name()) return false;
    }

    if (n1->IsVar()) {
      const auto &var1 = n1->Var();
      const auto &var2 = n2->Var();
      if ((var1 == nullptr) != (var2 == nullptr)) return false;
    }

    if (n1->IsOp()) {
      const auto &op1 = n1->Op();
      const auto &op2 = n2->Op();
      if ((op1 == nullptr) != (op2 == nullptr)) return false;

      const auto &op1_input = op1->InputNames();
      const auto &op2_input = op2->InputNames();
      if (op1_input.size() != op2_input.size()) return false;
      if (op1_input != op2_input) return false;

      for (size_t i = 0; i < op1_input.size(); ++i) {
        if (op1->Input(op1_input[i]) != op2->Input(op2_input[i])) return false;
      }

      const auto &op1_output = op1->OutputNames();
      const auto &op2_output = op2->OutputNames();
      if (op1_output.size() != op2_output.size()) return false;
      if (op1_output != op2_output) return false;

      for (size_t i = 0; i < op1_output.size(); ++i) {
        if (op1->Output(op1_output[i]) != op2->Output(op2_output[i]))
          return false;
      }
    }
  }
  return true;
}

inline bool CheckGraphSame(ir::Graph *g1, ir::Graph *g2) {
  if (g1 == nullptr || g2 == nullptr) return true;

  if (FLAGS_convert_all_blocks) {
    if (g1->SubGraphsSize() != g2->SubGraphsSize()) return false;

    for (size_t i = 0; i < g1->SubGraphsSize(); ++i) {
      if (!CheckSubGraphSame(g1->GetSubGraph(i), g2->GetSubGraph(i)))
        return false;
    }
  } else {
    if (!CheckSubGraphSame(g1, g2)) return false;
  }
  return true;
}

TEST(BuildStrategy, Basic) {
  BuildStrategy build_strategy;

  ProgramDesc prog;
  ir::Graph old_graph(prog), graph(prog);

  BuildStrategyApply(&build_strategy, &graph);

  ASSERT_TRUE(CheckGraphSame(&old_graph, &graph));
}

TEST(BuildStrategy, TestSingleGraph) {
  BuildStrategy build_strategy;
  auto graph = CreateGraph();
  ir::Graph old_graph(graph->OriginProgram());

  BuildStrategyApply(&build_strategy, graph.get());

  // graph should not change for no pass here
  ASSERT_TRUE(CheckGraphSame(&old_graph, graph.get()));
}

TEST(BuildStrategy, TestMultiGraph) {
  // Set FLAGS_convert_all_blocks to true to make sure this test works.
  bool flag_temp = FLAGS_convert_all_blocks;
  FLAGS_convert_all_blocks = true;

  BuildStrategy build_strategy;
  auto graph = CreateMultiGraph();
  ir::Graph old_graph(graph->OriginProgram());

  BuildStrategyApply(&build_strategy, graph.get());

  // graph should not change for no pass here
  ASSERT_TRUE(CheckGraphSame(&old_graph, graph.get()));

  // Recover FLAGS_convert_all_blocks.
  FLAGS_convert_all_blocks = flag_temp;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
