// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/fc_mkldnn_pass.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FCMKLDNNPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  Init("fc_mkldnn_pass", graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("fc_mkldnn_pass/x")
                ->AsInput()
                ->assert_is_op_input("fc", "Input");
  patterns::FCMKLDNN fc_pattern(gpd.mutable_pattern(), "fc_mkldnn_pass");
  fc_pattern(x, true /*with bias*/);

  int found_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Handle FC MKL-DNN pass";
    if (!(graph->Has("use_mkldnn") && graph->Get<bool>("use_mkldnn"))) {
      VLOG(3) << "do not perform fc fuse";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(output, output, fc_pattern);

    OpDesc* desc = fc->Op();
    auto dims = fc->inputs[0]->Var()->GetShape();
    auto dim_num = dims.size();
    bool are_dims_supported = dim_num >= 2 && dim_num <= 4;
    constexpr size_t height_axis = 2;
    constexpr size_t width_axis = 3;
    bool is_size_supported =
        dim_num == 4 ? (dims[width_axis] == 1 && dims[height_axis] == 1) : true;
    if (!are_dims_supported || !is_size_supported) {
      VLOG(3) << "Do not enable FC MKL-DNN for dimensions different than"
                 "2, 3 & 4, or when width or height is different than one.";
      return;
    }
    desc->SetAttr("use_mkldnn", true);

    found_fc_count++;
  };

  gpd(graph, handler);

  AddStatis(found_fc_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_mkldnn_pass, paddle::framework::ir::FCMKLDNNPass);
