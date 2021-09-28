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

#include "paddle/fluid/framework/ir/remove_c_identity_pass.h"

#include <cmath>
#include <string>
#include "paddle/fluid/framework/op_proto_maker.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

RemoveCIdentityPass::RemoveCIdentityPass() {}

void RemoveCIdentityPass::ApplyImpl(ir::Graph* graph) const {
  std::cout << "========> remove c_identity_pass applyImpl" << std::endl;
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "remove_c_identity_pass";
  FusePassBase::Init(name_scope, graph);

  GraphPatternDetector gpd;
  patterns::CIdentity c_identity_pattern(gpd.mutable_pattern(), name_scope);
  c_identity_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "remove c_identity pass";
    std::cout << "==========> remove c_identity_pass handler" << std::endl;
    GET_IR_NODE_FROM_SUBGRAPH(c_identity_op, c_identity_op, c_identity_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(c_identity_in_x, c_identity_in_x,
                              c_identity_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(c_identity_out, c_identity_out,
                              c_identity_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, c_identity_pattern);

    auto* next_op_desc = next_op->Op();
    auto next_op_inputs = next_op_desc->InputNames();
    for (const auto& name : next_op_inputs) {
      auto input_names = next_op_desc->Input(name);
      std::replace(input_names.begin(), input_names.end(),
                   c_identity_out->Name(), c_identity_in_x->Name());
      next_op_desc->SetInput(name, input_names);
    }
    IR_NODE_LINK_TO(c_identity_in_x, next_op);

    GraphSafeRemoveNodes(graph, {c_identity_op, c_identity_out});
    ++found_count;
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(remove_c_identity_pass,
              paddle::framework::ir::RemoveCIdentityPass);
