#include "paddle/fluid/framework/ir/conv_elementwise_add_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);

  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", true);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetOutput("Output", outputs);
  } else if (type == "elementwise_add") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", outputs);
  }
}

TEST(ConvElementwiseAddMKLDNNFusePass, ConvolutionWithElementwiseAddWithOps) {
  auto build_program_desc = [&]() -> ProgramDesc {
    ProgramDesc prog;
    for (auto& v :
      std::vector<std::string>({"a", "b", "weights", "c", "d", "e", "f", "g"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v == "weights" || v == "bias") {
        var->SetPersistable(true);
      }
    }
  
    SetOp(&prog, "OP0", {"a"}, {"b"});
    SetOp(&prog, "OP1", {"c"}, {"d"});
    SetOp(&prog, "conv2d", {"b", "weights"}, {"e"});
    SetOp(&prog, "elementwise_add", {"e", "d"}, {"f"});
    SetOp(&prog, "OP3", {"f"}, {"g"});

    return prog;
  };

  auto prog = build_program_desc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();
 
  EXPECT_EQ(original_nodes_num - 4 + 1, current_nodes_num);
  // Assert conv_relu op in newly generated graph
  int conv_count = 0;
  int elementwise_add_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      ++conv_count;
    }
    if (node->IsOp() && node->Op()->Type() == "elementwise_add") {
      ++elementwise_add_count;
    }
    /*
      if (node->Op()->HasAttr("use_mkldnn")) {
        bool use_mkldnn = boost::get<bool>(node->Op()->GetAttr("use_mkldnn"));
        if (use_mkldnn) {
          if (node->Op()->HasAttr("fuse_sum")) {
//            bool fuse_sum = boost::get<bool>(node->Op()->GetAttr("fuse_sum"));
            if (fuse_sum) {
              ++conv_elementwise_add_count;
            }
          }
        }
      }
    }
    */
  }
  EXPECT_EQ(conv_count, 1);
  EXPECT_EQ(elementwise_add_count, 0);
}

TEST(ConvElementwiseAddMKLDNNFusePass, OnlyConvolutionElementwiseAdd) {
  auto build_program_desc = [&]() -> ProgramDesc {
    ProgramDesc prog;
    for (auto& v :
      std::vector<std::string>({"a", "b", "weights"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::LOD_TENSOR);
      if (v == "weights" || v == "bias") {
        var->SetPersistable(true);
      }
    }
  
    SetOp(&prog, "conv2d", {"a", "weights"}, {"b"});
    SetOp(&prog, "elementwise_add", {"b", "c"}, {"d"});

    return prog;
  };

  auto prog = build_program_desc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();
 
  EXPECT_EQ(original_nodes_num - 4 + 1, current_nodes_num);
  // Assert conv_relu op in newly generated graph
  int conv_count = 0;
  int elementwise_add_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      ++conv_count;
    }
    if (node->IsOp() && node->Op()->Type() == "elementwise_add") {
      ++elementwise_add_count;
    }
    /*
      if (node->Op()->HasAttr("use_mkldnn")) {
        bool use_mkldnn = boost::get<bool>(node->Op()->GetAttr("use_mkldnn"));
        if (use_mkldnn) {
          if (node->Op()->HasAttr("fuse_sum")) {
//            bool fuse_sum = boost::get<bool>(node->Op()->GetAttr("fuse_sum"));
            if (fuse_sum) {
              ++conv_elementwise_add_count;
            }
          }
        }
      }
    }
    */
  }
  EXPECT_EQ(conv_count, 1);
  EXPECT_EQ(elementwise_add_count, 0);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
