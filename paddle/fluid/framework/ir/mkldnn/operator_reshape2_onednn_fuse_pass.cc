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

#include "paddle/fluid/framework/ir/mkldnn/operator_reshape2_onednn_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseOperatorReshape2OneDNNPass::ApplyImpl(Graph *graph) const {
  // THIS FUSE WILL WORK ONLY WITH OPERATORS THAT OUTPUTS PLAIN MEMORY, F.E.
  // ABCD FOR 4D! BE AWARE OF THAT!
  std::vector<std::pair<std::string, int>> ops_and_outputs = {
      {"fc", 1}, {"transpose2", 2}};

  for (const auto &op_and_outputs : ops_and_outputs)
    FuseReshape2(graph, op_and_outputs.first, op_and_outputs.second);
}

void FuseOperatorReshape2OneDNNPass::FuseReshape2(Graph *graph,
                                                  const std::string &op_type,
                                                  int num_of_outputs) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(op_type + "_reshape2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorReshape2 op_reshape2_pattern(
      gpd.mutable_pattern(), op_type + "_reshape2_onednn_fuse_pass");
  op_reshape2_pattern(op_type, num_of_outputs);

  int found_operator_reshape2_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    GET_IR_NODE_FROM_SUBGRAPH(operator_op, preceding_op, op_reshape2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        operator_out, preceding_op_out, op_reshape2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_op, reshape2_op, op_reshape2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_out, reshape2_out, op_reshape2_pattern);

    if (!operator_op->Op()->HasAttr("use_mkldnn") ||
        (operator_op->Op()->HasAttr("use_mkldnn") &&
         !(PADDLE_GET_CONST(bool, operator_op->Op()->GetAttr("use_mkldnn"))))) {
      VLOG(4) << "Only oneDNN version of " << op_type
              << "can be fused with reshape2.";
      return;
    }

    if (operator_op->Op()->HasAttr("fused_unsqueeze2_axes")) {
      VLOG(4) << "Cannot do " << op_type << " + reshape2 fuse, because "
              << op_type << " is already fused with unsqueeze2!";
      return;
    }

    std::vector<int> reshape2_shape =
        PADDLE_GET_CONST(std::vector<int>, reshape2_op->Op()->GetAttr("shape"));

    int num_of_minus_ones = 0;

    for (size_t i = 0; i < reshape2_shape.size(); ++i) {
      if (reshape2_shape[i] == 0) {
        VLOG(4) << "OneDNN op+reshape2 fuse pass does not support zero dims, "
                   "skipping";
        return;
      } else if (reshape2_shape[i] == -1) {
        ++num_of_minus_ones;
      }
    }

    if (num_of_minus_ones > 1) {
      VLOG(4) << "Number of -1 values inside of reshape2 shouldn't be greater "
                 "than one in op+reshape2 oneDNN fuse pass, skipping";
      return;
    }

    auto const &names = reshape2_op->Op()->InputNames();

    bool has_shape_tensor =
        std::find(names.begin(), names.end(), "ShapeTensor") != names.end();
    bool has_shape_tensor_list =
        std::find(names.begin(), names.end(), "ShapeTensorList") != names.end();

    if (has_shape_tensor &&
        reshape2_op->Op()->Input("ShapeTensor").size() > 0) {
      VLOG(4) << "Cannot fuse " << op_type
              << " and reshape2 because reshape2 dims are specified by "
                 "ShapeTensor!";
      return;
    }

    if (has_shape_tensor_list &&
        reshape2_op->Op()->Input("ShapeTensorList").size() > 0) {
      VLOG(4) << "Cannot fuse " << op_type
              << " and reshape2 because reshape2 dims are specified by "
                 "ShapeTensorList!";
      return;
    }

    operator_op->Op()->SetAttr("fused_reshape2_shape", reshape2_shape);
    operator_op->Op()->SetOutput("Out", {reshape2_out->Name()});

    IR_OP_VAR_LINK(operator_op, reshape2_out);
    GraphSafeRemoveNodes(g, {reshape2_op, operator_out});
    found_operator_reshape2_count++;
  };

  gpd(graph, handler);
  AddStatis(found_operator_reshape2_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_operator_reshape2_count > 0)
    PrettyLogDetail("---    fused %d %s with reshape2",
                    found_operator_reshape2_count,
                    op_type);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(operator_reshape2_onednn_fuse_pass,
              paddle::framework::ir::FuseOperatorReshape2OneDNNPass);
REGISTER_PASS_CAPABILITY(operator_reshape2_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("reshape2", 0)
            .GE("fc", 0));
