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

#include "paddle/fluid/framework/ir/fc_mkldnn_pass.h"
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void Transpose(float* data, size_t rows, unsigned cols) {
  std::vector<float> temp(rows * cols);

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      temp[j * rows + i] = data[i * cols + j];
    }
  }

  std::copy(temp.begin(), temp.end(), data);
}

std::unique_ptr<ir::Graph> FCMKLDNNPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());
  FusePassBase::Init("fc_mkldnn_pass", graph.get());

  auto* scope = param_scope();
  PADDLE_ENFORCE(scope);

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
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(output, output, fc_pattern);

    FuseOptions fuse_option = FindFuseOption(*fc, *fc);
    if (fuse_option == DO_NOT_FUSE || fuse_option == FUSE_NATIVE) {
      VLOG(3) << "do not perform fc fuse";
      return;
    }

    OpDesc* desc = fc->Op();
    auto in_size = fc->inputs[0]->Var()->GetShape().size();
    if (in_size != 2 && in_size != 4) {    // if input dimensions aren't
      desc->SetAttr("use_mkldnn", false);  // supported don't perform the fuse
      return;
    }

    auto* weights_var = scope->FindVar(weights->Name());
    auto* weights_tensor = weights_var->GetMutable<LoDTensor>();
    auto dims = weights_tensor->dims();
    auto rows = dims[0];
    auto cols = dims[1];
    auto data = weights_tensor->mutable_data<float>(platform::CPUPlace());
    Transpose(data, rows, cols);

    PADDLE_ENFORCE(subgraph.count(x));

    found_fc_count++;
  };

  gpd(graph.get(), handler);

  AddStatis(found_fc_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_mkldnn_pass, paddle::framework::ir::FCMKLDNNPass);
