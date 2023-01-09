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

#include "paddle/fluid/framework/ir/mkldnn/interpolate_mkldnn_pass.h"

#include <string>
#include <vector>

#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Graph;

void InterpolateOneDNNPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  if (!(graph->Has("use_mkldnn") && graph->Get<bool>("use_mkldnn"))) {
    VLOG(3) << "Do not handle interpolate_mkldnn_pass";
    return;
  }
  VLOG(4) << "Handle interpolate_mkldnn_pass";

  Init("interpolate_mkldnn_pass", graph);

  int found_count = 0;
  const std::vector<std::string> interpolate_op_types = {"bilinear_interp",
                                                         "nearest_interp",
                                                         "trilinear_interp",
                                                         "bicubic_interp",
                                                         "linear_interp",
                                                         "bilinear_interp_v2",
                                                         "nearest_interp_v2"};

  for (const Node* node : graph->Nodes()) {
    if (node->IsOp() && std::find(interpolate_op_types.begin(),
                                  interpolate_op_types.end(),
                                  node->Name()) != interpolate_op_types.end()) {
      auto* op_desc = node->Op();
      op_desc->SetAttr("use_mkldnn", true);
      ++found_count;
    }
  }

  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(interpolate_mkldnn_pass,
              paddle::framework::ir::InterpolateOneDNNPass);
