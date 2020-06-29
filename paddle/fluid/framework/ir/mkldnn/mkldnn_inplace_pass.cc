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
  std::unordered_set<std::string> inplaced_vars;
  GraphPatternDetector gpd;
  patterns::MKLDNNInPlace mkldnn_inplace{gpd.mutable_pattern(),
                                         "mkldnn_inplace"};
  mkldnn_inplace();

  int found_inplace_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(3) << "Start to handle MKL-DNN In-Place pass";

    GET_IR_NODE_FROM_SUBGRAPH(current_op, inplace_to_be_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(current_op_in, inplace_to_be_op_in,
                              mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(current_op_out, inplace_to_be_op_out,
                              mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, mkldnn_inplace);
    GET_IR_NODE_FROM_SUBGRAPH(next_op_out, next_op_out, mkldnn_inplace);

    if ((current_op->Op()->HasAttr("use_mkldnn") == false) ||
        (BOOST_GET_CONST(bool, current_op->Op()->GetAttr("use_mkldnn")) ==
         false)) {
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

    VLOG(3) << "DNNL Inplace op(" << current_op->id() << ") "
            << "Curr Node In: " << current_op_in->Name()
            << " Curr Node out: " << current_op_out->Name();

    VLOG(3) << "DNNL Inplace next op(" << next_op->id() << ") "
            << " next Node out: " << next_op_out->Name();

    auto inputs = current_op->Op()->Inputs();
    auto outputs = current_op->Op()->Outputs();
    auto in_to_outs = infer_inplace(false);  // strictly no CUDA for MKL-DNN
    VLOG(3) << "DNNL InplaceInferer op(" << current_op->id() << ") "
            << in_to_outs.begin()->first << ": "
            << inputs[in_to_outs.begin()->first][0] << " "
            << in_to_outs.begin()->second << ": "
            << outputs[in_to_outs.begin()->second][0];
    // If InferInplace pattern does not contain input node then skip
    auto inplace_input_vec = inputs[in_to_outs.begin()->first];
    if (std::find(inplace_input_vec.begin(), inplace_input_vec.end(),
                  current_op_in->Name()) == inplace_input_vec.end()) {
      VLOG(3) << "DNNL in-place pass SKIP pattern ";
      return;
    }

    // Checking if this particular node (to be inplaced, overwritten)
    // is used anywhere else apart from inplaced op
    auto input_consumers = current_op_in->outputs;
    if (input_consumers.size() > 1) {
      VLOG(3) << "DNNL in-place pass FAIL: in-place var cannot "
                 "be an input to multiple operators";
      return;
    } else {
      // We will prevent in-place when
      // input is used in other part of graph, unless it was a result of
      // inplacing
      // Allow to next op out reuse inpuit var, as this is the same chaing
      if (inplaced_vars.find(current_op_in->Name()) == inplaced_vars.end()) {
        for (const Node* n : graph->Nodes()) {
          if ((n->id() != current_op_in->id()) &&
              (n->id() != next_op_out->id()) &&
              (n->Name() == current_op_in->Name())) {
            VLOG(3) << "DNNL in-place pass FAIL var used in diffrent part of "
                       "graph ";
            return;
          }
        }
      }
    }

    // If this op was alrady inplaced in previous pass placements
    // then we need to update input of next op
    // but original name to be changed is gone, so we need to remember it
    // on first time given op is to be inplaced
    if (current_op_in->Name() != current_op_out->Name()) {
      original_output_names[current_op->Name() + current_op_in->Name()] =
          current_op_out->Name();
    } else {
      VLOG(3) << "DNNL Inplace: Current op already inplaced! ";
    }

    // It may be that next op is reusing some of vars, we need to
    // make sure that unwanted inplace is not created
    for (auto& n : current_op_out->outputs) {
      auto& n_op_infer_inplace =
          OpInfoMap::Instance().Get(n->Op()->Type()).infer_inplace_;
      if ((n_op_infer_inplace == nullptr)) {
        for (auto& m : n->outputs) {
          if (m->Name() == current_op_in->Name()) {
            VLOG(3) << "DNNL in-place pass FAIL: in-place var cannot "
                       "be an output to non-inplaced next op";
            return;
          }
        }
      }
    }

    auto original_name =
        original_output_names[current_op->Name() + current_op_in->Name()];
    current_op_out->RenameVar(current_op_in->Name());

    // Get mapping of input to output
    auto out_name = in_to_outs.begin()->second;
    current_op->Op()->SetOutput(
        out_name, std::vector<std::string>({current_op_out->Name()}));
    // Record var name
    inplaced_vars.insert(current_op_out->Name());

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
      // for variable we changed (original name)
      // TODO(jczaja): make recursive propagation of inplace
      auto next_op_inplace_inputs = inputs[in_to_outs.begin()->first];
      if ((next_op_inplace_inputs == outputs[in_to_outs.begin()->second]) &&
          (std::find(next_op_inplace_inputs.begin(),
                     next_op_inplace_inputs.end(),
                     original_name) != next_op_inplace_inputs.end())) {
        VLOG(3) << "DNNL InPlace: Next Op is in-placed , updating its "
                   "input "
                   "and output var!";
        next_op->Op()->SetOutput(
            out_name, std::vector<std::string>({current_op_out->Name()}));
        next_op_out->RenameVar(current_op_in->Name());
        // Get ops that next_op_out is linked to and update their input
        auto next_op_out_consumers = next_op_out->outputs;  // Has to be ops
        for (auto& c : next_op_out_consumers) {
          c->Op()->RenameInput(original_name, current_op_out->Name());
        }
      }
    }

    next_op->Op()->RenameInput(original_name, current_op_out->Name());

    found_inplace_count++;
    VLOG(3) << "DNNL InPlace applied!";
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mkldnn_inplace_pass, paddle::framework::ir::MKLDNNInPlacePass);
