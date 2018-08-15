#include "paddle/fluid/framework/ir/fc_fuse_pass.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("Xs", inputs);
  op->SetOutput("Xs", outputs);
}

// a->OP0->b
// a->OP1->c
// (b, c)->mul->d
// (d, e)->elementwise_add->f
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a", "b", "c", "d", "e", "f"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "c") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", std::vector<std::string>({"a"}),
        std::vector<std::string>({"c"}));
  SetOp(&prog, "mul", std::vector<std::string>({"b", "c"}),
        std::vector<std::string>({"d"}));
  SetOp(&prog, "elementwise_add", std::vector<std::string>({"d", "e"}),
        std::vector<std::string>({"f"}));

  return prog;
}

TEST(FCFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("fc_fuse_pass");

  int pre_nodes = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int after_nodes = graph->Nodes().size();

  // Remove 3 Nodes: MUL,ELEMENTWISE_ADD, mul_out
  // Add 1 Node: FC
  EXPECT_EQ(pre_nodes - 2, after_nodes);

  // Assert fc op in newly generated graph
  int fc_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      ++fc_count;
    }
  }
  EXPECT_EQ(fc_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_fuse_pass);
