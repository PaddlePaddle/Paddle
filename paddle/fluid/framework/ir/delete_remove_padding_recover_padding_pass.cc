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

#include "paddle/fluid/framework/ir/delete_remove_padding_recover_padding_pass.h"

#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle::framework::ir::patterns {

void RecoverPadding::operator()() {
  // Create nodes for recover_padding.
  auto *recover_padding_input =
      pattern->NewNode(recover_padding_input_repr())
          ->assert_is_op_input("recover_padding", "Input");
  auto *recover_padding_op = pattern->NewNode(recover_padding_op_repr())
                                 ->assert_is_op("recover_padding");
  auto *recover_padding_out =
      pattern->NewNode(recover_padding_out_repr())
          ->assert_is_op_output("recover_padding", "Out");

  // Add links for recover_padding op.
  recover_padding_op->LinksFrom({recover_padding_input})
      .LinksTo({recover_padding_out});
}
}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

void DeleteRemovePaddingRecoverPaddingPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init(name_scope_, graph);
  int found_subgraph_count = 0;

  //
  GraphPatternDetector gpd;
  patterns::RecoverPadding recover_padding(
      gpd.mutable_pattern(), "delete_remove_padding_recover_padding_pass");
  recover_padding();

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    VLOG(3) << "delete_remove_padding_recover_padding_pass";

    GET_IR_NODE_FROM_SUBGRAPH(
        recover_padding_input, recover_padding_input, recover_padding);
    GET_IR_NODE_FROM_SUBGRAPH(
        recover_padding_op, recover_padding_op, recover_padding);
    GET_IR_NODE_FROM_SUBGRAPH(
        recover_padding_out, recover_padding_out, recover_padding);

    std::unordered_set<const Node *> del_node_set;

    bool delete_recover_padding = true;
    for (size_t i = 0; i < recover_padding_out->outputs.size();  // NOLINT
         ++i) {
      if (recover_padding_out->outputs[i]->Name() ==
          "remove_padding") {  // op_node
        auto *remove_padding_out_node =
            recover_padding_out->outputs[i]->outputs[0];  // NOLINT // var_node
        auto *out_op_node =
            remove_padding_out_node->outputs[0];  // NOLINT // op_node
        IR_NODE_LINK_TO(recover_padding_input, out_op_node);
        del_node_set.insert(recover_padding_out->outputs[i]);  // NOLINT
        del_node_set.insert(remove_padding_out_node);
        out_op_node->Op()->RenameInput(remove_padding_out_node->Name(),
                                       recover_padding_input->Name());
        found_subgraph_count++;
      } else {
        delete_recover_padding = false;
      }
    }
    if (delete_recover_padding) {
      del_node_set.insert(recover_padding_op);
      del_node_set.insert(recover_padding_out);
    }
    GraphSafeRemoveNodes(graph, del_node_set);
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(delete_remove_padding_recover_padding_pass,
              paddle::framework::ir::DeleteRemovePaddingRecoverPaddingPass);
