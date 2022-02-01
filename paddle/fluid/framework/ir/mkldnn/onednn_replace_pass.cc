// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/onednn_replace_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

#include <unordered_map>

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void OneDNNReplacePass::ApplyImpl(Graph *graph) const {
  const std::unordered_map<std::string, std::vector<std::string>>
      elementwise_io_map = {{"Inputs", {"X", "Y"}}, {"Outputs", {"Out"}}};
  const std::unordered_map<
      std::string, std::unordered_map<std::string, std::vector<std::string>>>
      in_out_map = {{"elementwise_add", elementwise_io_map},
                    {"elementwise_mul", elementwise_io_map},
                    {"elementwise_sub", elementwise_io_map}};

  for (const auto &op_to_replace : in_out_map) {
    GraphPatternDetector gpd = GraphPatternDetector();
    auto op_type = op_to_replace.first;
    std::vector<PDNode *> inputs;
    auto input_names = (op_to_replace.second).at("Inputs");
    for (const auto &input : input_names)
      inputs.emplace_back(
          gpd.mutable_pattern()->NewNode()->AsInput()->assert_is_op_input(
              op_type, input));

    std::vector<PDNode *> outputs;
    auto output_names = (op_to_replace.second).at("Outputs");
    for (const auto &output : output_names)
      outputs.emplace_back(
          gpd.mutable_pattern()->NewNode()->AsOutput()->assert_is_op_output(
              op_type, output));

    patterns::OneDNNOp onednn_op_pattern(gpd.mutable_pattern(),
                                         op_type + "_onednn");
    onednn_op_pattern(inputs, outputs, op_type);

    auto handler = [op_type, onednn_op_pattern](
        const GraphPatternDetector::subgraph_t &subgraph, Graph *g) {
      VLOG(4) << "Replace " << op_type << " op with onednn version.";
      // ops
      GET_IR_NODE_FROM_SUBGRAPH(op_to_replace, op_to_replace,
                                onednn_op_pattern);

      auto *replace_op = op_to_replace->Op();

      replace_op->SetType(op_type + "_one_dnn");
    };

    gpd(graph, handler);
  }

  // get_node_from_factory();
  // replace_node();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(onednn_replace_pass, paddle::framework::ir::OneDNNReplacePass);
REGISTER_PASS_CAPABILITY(one_dnn_replace_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .LE("elementwise_sub", 1)
            .LE("elementwise_mul", 1));
