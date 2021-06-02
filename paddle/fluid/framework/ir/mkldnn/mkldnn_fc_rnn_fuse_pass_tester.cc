// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/fc_gru_fuse_pass_tester.h"
#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass_tester.h"
#include "paddle/fluid/framework/ir/mkldnn/mkldnn_placement_pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void TestFcRNNFusePass(const std::string& pass_name) {
  std::unique_ptr<ir::Graph> graph =
      (pass_name == "fc_gru_fuse_pass" ? fc_gru_test::PrepareGraph()
                                       : fc_lstm_test::PrepareGraph());
  auto mkldnn_placement_pass_ =
      PassRegistry::Instance().Get("mkldnn_placement_pass");
  mkldnn_placement_pass_->Set("mkldnn_enabled_op_types",
                              new std::unordered_set<std::string>({}));
  graph->Set("__param_scope__", (pass_name == "fc_gru_fuse_pass"
                                     ? fc_gru_test::CreateParamScope()
                                     : fc_lstm_test::CreateParamScope()));
  graph.reset(mkldnn_placement_pass_->Apply(graph.release()));

  auto check_num_mkldnn_nodes = [&](const std::unique_ptr<ir::Graph>& graph) {
    int nodes_cout = 0;
    for (auto* node : graph->Nodes()) {
      if (node->IsOp()) {
        auto* op = node->Op();
        if (op->GetAttrIfExists<bool>("use_mkldnn")) nodes_cout++;
      }
    }
    return nodes_cout;
  };
  int num_mkldnn_nodes_before = check_num_mkldnn_nodes(graph);

  auto fc_rnn_fuse_pass_ = PassRegistry::Instance().Get(pass_name);
  graph.reset(fc_rnn_fuse_pass_->Apply(graph.release()));
  int num_mkldnn_nodes_after = check_num_mkldnn_nodes(graph);

  PADDLE_ENFORCE_EQ(num_mkldnn_nodes_before, num_mkldnn_nodes_after,
                    platform::errors::PreconditionNotMet(
                        "The number of nodes with \"use_mkldnn\" attr before "
                        "and after the fuse are not equal"));
}

TEST(FcGruFusePass, use_mkldnn) { TestFcRNNFusePass("fc_gru_fuse_pass"); }

TEST(FcLstmFusePass, use_mkldnn) { TestFcRNNFusePass("fc_lstm_fuse_pass"); }

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(mkldnn_placement_pass);
USE_PASS(fc_gru_fuse_pass);
USE_PASS(fc_lstm_fuse_pass);
