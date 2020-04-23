// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_inplace_pass.h"

#include <gtest/gtest.h>
#include <boost/logic/tribool.hpp>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/framework/op_registry.h"

USE_OP(softmax);
USE_OP_DEVICE_KERNEL(softmax, MKLDNN);
USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, MKLDNN);
USE_OP(relu);

namespace paddle {
namespace framework {
namespace ir {

class MKLDNNInplacePassTest {
 private:
  void SetOp(ProgramDesc* prog, const std::string& type,
             const std::string& name, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs,
             boost::tribool use_mkldnn) {
    auto* op = prog->MutableBlock(0)->AppendOp();

    op->SetType(type);

    if (!boost::indeterminate(use_mkldnn))
      op->SetAttr("use_mkldnn", use_mkldnn);

    if (type == "conv2d") {
      op->SetAttr("name", name);
      op->SetInput("Input", {inputs[0]});
      op->SetInput("Filter", {inputs[1]});
      op->SetInput("Bias", {inputs[2]});
    } else if (type == "relu") {
      op->SetInput("X", inputs);
    } else if (type == "softmax") {
      op->SetAttr("axis", -1);
      op->SetInput("X", inputs);
    } else if (type == "elementwise_add") {
      op->SetInput("X", {inputs[0]});
      op->SetInput("Y", {inputs[1]});
    } else {
      FAIL() << "Unexpected operator type.";
    }
    op->SetOutput("Out", {outputs[0]});
  }

  ProgramDesc BuildProgramDesc(const std::string& mkldnn_enabled_op,
                               bool branched) {
    ProgramDesc prog;

    for (auto& v :
         std::vector<std::string>({"a", "weights", "bias", "f", "g", "h", "i",
                                   "j", "k", "l", "m", "z"})) {
      auto* var = prog.MutableBlock(0)->Var(v);
      var->SetType(proto::VarType::SELECTED_ROWS);
      if (v == "weights" || v == "bias") {
        var->SetPersistable(true);
      }
    }

    SetOp(&prog, "conv2d", "conv1",
          std::vector<std::string>({"a", "weights", "bias"}),
          std::vector<std::string>({"f"}), boost::indeterminate);
    SetOp(&prog, "relu", "relu1", std::vector<std::string>({"f"}),
          std::vector<std::string>({"g"}),
          mkldnn_enabled_op.compare("relu") == 0);
    SetOp(&prog, "softmax", "softmax1", std::vector<std::string>({"g"}),
          std::vector<std::string>({"h"}),
          mkldnn_enabled_op.compare("softmax") == 0);
    SetOp(&prog, "elementwise_add", "elementwise_add1",
          std::vector<std::string>({"h", "i"}), std::vector<std::string>({"j"}),
          mkldnn_enabled_op.compare("elementwise_add") == 0);
    SetOp(&prog, "relu", "relu2", std::vector<std::string>({"j"}),
          std::vector<std::string>({"k"}),
          mkldnn_enabled_op.compare("softmax") == 0);
    if (branched == true) {
      SetOp(&prog, "softmax", "softmax2", std::vector<std::string>({"g"}),
            std::vector<std::string>({"z"}),
            mkldnn_enabled_op.compare("softmax") == 0);
    }

    return prog;
  }

 public:
  void MainTest(const std::string& mkldnn_enabled_op, bool branched,
                unsigned expected_use_mkldnn_true_count) {
    auto prog = BuildProgramDesc(mkldnn_enabled_op, branched);

    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    auto pass = PassRegistry::Instance().Get("mkldnn_inplace_pass");

    graph.reset(pass->Apply(graph.release()));

    unsigned use_mkldnn_true_count = 0;
    std::unordered_map<std::string, std::string> input_names;
    std::unordered_map<std::string, std::string> output_names;

    input_names["softmax"] = "X";
    output_names["softmax"] = "Out";
    input_names["elementwise_add"] = "X";
    output_names["elementwise_add"] = "Out";

    VLOG(3) << DebugString(graph);

    for (auto* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto* op = node->Op();
        if (op->Type() == mkldnn_enabled_op) {
          auto ins = op->Inputs();
          auto outs = op->Outputs();
          // Input and output are the same var
          if (ins[input_names[mkldnn_enabled_op]] ==
              outs[output_names[mkldnn_enabled_op]]) {
            ++use_mkldnn_true_count;
          }
        }
      }
    }

    EXPECT_EQ(use_mkldnn_true_count, expected_use_mkldnn_true_count);
  }
};

TEST(MKLDNNInplacePass, inplace_softmax) {
  // softmax to be mkl-dnn enabled and made in-place
  MKLDNNInplacePassTest().MainTest("softmax", false, 1);
}

TEST(MKLDNNInplacePass, inplace_softmax_branched) {
  // softmax's input is shared by two branches. so no in-place
  MKLDNNInplacePassTest().MainTest("softmax", true, 0);
}

TEST(MKLDNNInplacePass, inplace_elementwise_add) {
  // Two elementwise_add mkl-dnn enabled op instances to be made inplace
  MKLDNNInplacePassTest().MainTest("elementwise_add", false, 1);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(mkldnn_inplace_pass);
