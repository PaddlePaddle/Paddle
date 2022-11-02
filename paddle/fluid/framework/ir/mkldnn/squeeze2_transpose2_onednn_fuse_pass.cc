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
#include "paddle/fluid/framework/ir/mkldnn/squeeze2_transpose2_onednn_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/string/pretty_log.h"


namespace paddle {
namespace framework {
namespace ir{

using string::PrettyLogDetail;

void FuseSqueeze2Transpoe2OneDNNPass::ApplyImpl(Graph *graph) const{
  std::vector<std::pair<std::string, int>> ops_and_outputs = { {"elementwise_add", 1} };

  for (const auto &op_and_outputs : ops_and_outputs)
    FuseSqueeze2(graph, op_and_outputs.first, op_and_outputs.second);
}

void FuseSqueeze2Transpoe2OneDNNPass::FuseSqueeze2(Graph *graph, const std::string &op_type, int num_of_outputs) const{
  PADDLE_ENFORCE_NOT_NULL(graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));

  FusePassBase::Init(op_type + "_squeeze2_transpose2_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::Squeeze2Transpose2 squeeze2_transpose2_pattern(
    gpd.mutable_pattern(), op_type + "_squeeze2_transpose2_onednn_fuse_pass");
  squeeze2_transpose2_pattern(op_type, num_of_outputs);

  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph, Graph *g){
    GET_IR_NODE_FROM_SUBGRAPH(preceding_op, preceding_op, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        preceding_op_out, preceding_op_out, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(squeeze2_op, squeeze2_op, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        squeeze2_op_out, squeeze2_op_out, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_op, transpose2_op, squeeze2_transpose2_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        transpose2_op_out, transpose2_op_out, squeeze2_transpose2_pattern);
    
     LOG(INFO) << "fused_squeeze2_axes: " << found_count;

    if (!transpose2_op->Op()->HasAttr("use_mkldnn") || (transpose2_op->Op()->HasAttr("use_mkldnn") && !(PADDLE_GET_CONST(bool, transpose2_op->Op()->GetAttr("use_mkldnn"))))){
       VLOG(4) << "Only oneDNN version of transpose2 can be fused after with squeeze2.";
       return ;
    }


 LOG(INFO) << "fused_squeeze2_axes: " << found_count;

    auto const &names = squeeze2_op->Op()->InputNames();
    bool has_axes_tensor =
        std::find(names.begin(), names.end(), "AxesTensor") != names.end();
    bool has_axes_tensor_list =
        std::find(names.begin(), names.end(), "AxesTensorList") != names.end();
    if (has_axes_tensor &&
        squeeze2_op->Op()->Input("AxesTensor").size() > 0) {
      VLOG(4) << "Cannot fuse squeeze2 and transpose2 because squeeze2 dims are specified by "
              <<  "AxesTensor!";
      return;
    }

    if (has_axes_tensor_list &&
        squeeze2_op->Op()->Input("AxesTensorList").size() > 0) {
      VLOG(4) <<"Cannot fuse squeeze2 and transpose2 because squeeze2 dims are specified by "
              <<  "AxesTensorList!";
      return;
    }

 LOG(INFO) << "fused_squeeze2_axes: " << found_count;
 
    std::vector<int> squeeze2_axes = PADDLE_GET_CONST(std::vector<int>, squeeze2_op->Op()->GetAttr("axes"));
    transpose2_op->Op()->SetAttr("fused_squeeze2_axes", squeeze2_axes);
    transpose2_op->Op()->SetInput("X", {preceding_op_out->Name()});

    IR_VAR_OP_LINK(preceding_op_out, transpose2_op);
    GraphSafeRemoveNodes(g, {squeeze2_op, squeeze2_op_out});
    found_count++;

    LOG(INFO) << "fused_squeeze2_axes: " << found_count;
  };

  gpd(graph, handler);
  AddStatis(found_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) && found_count > 0){
    PrettyLogDetail("--- fused %d squeeze2 with transpose2", found_count);
  }
}

}
}
} // namespace paddle


REGISTER_PASS(squeeze2_transpose2_onednn_fuse_pass,
              paddle::framework::ir::FuseSqueeze2Transpoe2OneDNNPass);
REGISTER_PASS_CAPABILITY(squeeze2_transpose2_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .GE("squeeze2", 0)
            .GE("transpose2", 0));