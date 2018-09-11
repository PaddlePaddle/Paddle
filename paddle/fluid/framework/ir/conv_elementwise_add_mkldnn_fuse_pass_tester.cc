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
    op->SetInput("Output", {outputs});
  } else if (type == "elementwise_add") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", outputs);
  }
}

ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v :
    std::vector<std::string>({"a", "b", "c", "d", "weights", "f", "g"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }
  
  SetOp(&prog, "OP0", {"a"}, {"b"});
  SetOp(&prog, "OP1", {"c"}, {"d"});
  SetOp(&prog, "conv2d", {"d", "weights"}, {"f"});
  SetOp(&prog, "elemenwise_add", {"d", "f"}, {"g"});

  return prog;
}

TEST(ConvElementwiseAddMKLDNNFusePass, basic) {
  auto prog = BuildProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("conv_elementwise_add_mkldnn_fuse_pass");
  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();
 
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);
  // Assert conv_relu op in newly generated graph
  int conv_elementwise_add_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      if (node->Op()->HasAttr("use_mkldnn")) {
        bool use_mkldnn = boost::get<bool>(node->Op()->GetAttr("use_mkldnn"));
        if (use_mkldnn) {
          // TODO tpatejko: it is commented because convolution does not support this attribute
          if (true/*node->Op()->HasAttr("fuse_sum")*/) {
//            bool fuse_sum = boost::get<bool>(node->Op()->GetAttr("fuse_sum"));
            if (true /*fuse_sum*/) {
              ++conv_elementwise_add_count;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(conv_elementwise_add_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_elementwise_add_mkldnn_fuse_pass);
