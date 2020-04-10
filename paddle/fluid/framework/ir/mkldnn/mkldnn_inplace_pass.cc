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

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_inplace_pass.h"
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void MKLDNNInPlacePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  std::unordered_map<std::string, std::string> original_output_names;
  GraphPatternDetector gpd;
  patterns::MKLDNNInPlace mkldnn_inplace{gpd.mutable_pattern(),
                                         "mkldnn_inplace"};
  mkldnn_inplace();

  int found_inplace_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "Start to handle MKL-DNN In-Place pass";

    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(current_op, inplace_to_be_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(current_op_in, inplace_to_be_op_in,
                              mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(current_op_out, inplace_to_be_op_out,
                              mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(next_op_out, next_op_out, mkldnn_inplace);

    if ((current_op->Op()->HasAttr("use_mkldnn") == false) ||
        (boost::get<bool>(current_op->Op()->GetAttr("use_mkldnn")) == false)) {
      VLOG(3) << "do not perform mkl-dnn inplace: use_mkldnn missing or set to "
                 "false";
      return;
    }

    auto& infer_inplace =
        OpInfoMap::Instance().Get(current_op->Op()->Type()).infer_inplace_;
    if (!infer_inplace) {
      VLOG(3) << "do not perform mkl-dnn inplace: missing InplaceInferer";
      return;
    }

    auto count_specific_vars = [](VariableNameMap& mapvar,
                                  std::string& target_var_name) {
      unsigned int count = 0;
      for (auto& it : mapvar) {
        for (auto& var_name : it.second) {
          count += (var_name == target_var_name) ? 1 : 0;
        }
      }
      return count;
    };

    // Iterate over all nodes  that are ops
    // and check if in-place to be var is part of inputs
    // if positive then do not perform inplace
    for (const Node* n : graph->Nodes()) {
      if (n->IsOp()) {
        // Avoid searchin in op that is to be inplace
        if ((n->id() != current_op->id())) {
          auto* op = n->Op();
          auto inputs = op->Inputs();
          auto outputs = op->Outputs();
          auto in_place_input = current_op_in->Name();
          for (auto& it : inputs) {
            for (auto& var_name : it.second) {
              if (var_name == in_place_input) {
                if (n->id() == next_op->id()) {
                  // If next op is already having inplace var
                  // among its inputs then do not perform inplacing
                  if (count_specific_vars(inputs, var_name) > 0) {
                    VLOG(3)
                        << "MKL-DNN in-place pass FAIL: in-place var cannot "
                           "be an "
                           "input to next op in same chain";
                    return;
                  }
                } else if (n->id() == prev_op->id()) {
                  // Ok if op that is having current op input as its own
                  // input is directly before current op, and prev op
                  // is also in-place then we can in-placed current op
                  if (count_specific_vars(outputs, var_name) > 0) {
                    VLOG(3) << "MKL-DNN in-place pass OK: in-place var is "
                               "an input of prev op, but also of inplaced op.";
                  } else {
                    VLOG(3)
                        << "MKL-DNN in-place pass FAIL: in-place var cannot be "
                           "an "
                           "input to more than one operator of diffrent "
                           "branches";
                    return;
                  }
                } else {
                  // Neither prev_op or next_op
                  // If some no in pattern node is having our current input var
                  // among its
                  // inputs,  it is usually cycle, unless current op is already
                  // inplaced
                  if (current_op_in->Name() == current_op_out->Name()) {
                    VLOG(3)
                        << "MKL-DNN in-place pass OK: op was in-placed already,"
                           "but its output is used in multiple ops.";
                  } else {
                    VLOG(3)
                        << "MKL-DNN in-place pass FAIL: in-place var cannot be "
                           "an "
                           "input to more than one operator of diffrent "
                           "branches";
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }

    // If this op was alrady inplaced in previous pass placements
    // then we need to update input of next op
    // but original name to be changed is gone, so we need to remember it
    // on first time given op is to be inplaced
    if (current_op_in->Name() != current_op_out->Name()) {
      original_output_names[current_op->Name()] = current_op_out->Name();
    }
    auto original_name = original_output_names[current_op->Name()];
    current_op_out->RenameVar(current_op_in->Name());

    // Get mapping of input to output
    auto in_to_outs = infer_inplace(false);  // strictly no CUDA for MKL-DNN
    auto out_name = in_to_outs.begin()->second;
    current_op->Op()->SetOutput(
        out_name, std::vector<std::string>({current_op_out->Name()}));

    // If next op in a line is doing inplace
    // then we need to update its output as well

    // Get inferer of next op
    // If no inferer then we are done
    auto& next_op_infer_inplace =
        OpInfoMap::Instance().Get(next_op->Op()->Type()).infer_inplace_;
    if (next_op_infer_inplace) {
      auto in_to_outs = next_op_infer_inplace(false);
      auto out_name = in_to_outs.begin()->second;
      auto* op = next_op->Op();
      auto inputs = op->Inputs();
      auto outputs = op->Outputs();
      // Check if in-place happened
      if (inputs[in_to_outs.begin()->first] ==
          outputs[in_to_outs.begin()->second]) {
        VLOG(3) << "MKL-DNN InPlace: Next Op is in-placed , updating its input "
                   "and output var!";
        next_op->Op()->SetOutput(
            out_name, std::vector<std::string>({current_op_out->Name()}));
        next_op_out->RenameVar(current_op_in->Name());
      }
    }

    next_op->Op()->RenameInput(original_name, current_op_out->Name());

    found_inplace_count++;
    VLOG(3) << "MKL-DNN InPlace applied!";
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mkldnn_inplace_pass, paddle::framework::ir::MKLDNNInPlacePass);
