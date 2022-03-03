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

#pragma once
#include <iostream>
#include <vector>

#include "paddle/fluid/eager/api/to_static/run_program_op_node.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/eager/utils.h"

inline void run_program_dygraph_function(
    const std::vector<paddle::experimental::Tensor>& x,
    const std::vector<paddle::experimental::Tensor>& params,
    std::vector<paddle::experimental::Tensor*>& out,     // NOLINT
    std::vector<paddle::framework::Scope*>& step_scope,  // NOLINT
    std::vector<paddle::experimental::Tensor*>& dout,    // NOLINT
    const paddle::framework::AttributeMap& attrs) {
  std::cout << "start run run_program" << std::endl;
  // Call forward function
  RunProgramAPI(x, params, out, step_scope, dout, attrs);
  std::cout << "end run run_program" << std::endl;

  // Generate backward meta info.
  // auto p_autograd_in = egr::EagerUtils::unsafe_autograd_meta(x);
  // auto p_autograd_out = egr::EagerUtils::autograd_meta(&out);
  // bool trace_backward = egr::Controller::Instance().HasGrad();
  // bool require_any_grad =
  //     egr::EagerUtils::ComputeRequireGrad(trace_backward, p_autograd_in);

  // if (require_any_grad) {
  //   egr::EagerUtils::PassStopGradient(false /*generate_grad*/,
  //   p_autograd_out);

  //   p_autograd_out->SetSingleOutRankWithSlot(0, 0);

  //   // Init GradNode
  //   auto run_program_node = std::make_shared<GradNodeRunProgram>(/*
  //   fwd_in_slot_num */ 1,
  //                                                     /* bwd_in_slot_num */
  //                                                     1);

  //   // Set Next Edges
  //   run_program_node->AddEdges(p_autograd_in, /*slot id*/ 0);

  //   // Set TensorWrappers
  //   run_program_node->SetTensorWrappers_X({x});

  //   // Set Grad out rank as same as fwd input and set stop gradient to bwd
  //   run_program_node->SetGradOutMeta(p_autograd_in, /*slot id*/ 0);
  //   // Set Grad out rank as same as fwd input and set stop gradient to bwd
  //   run_program_node->SetGradInMeta(p_autograd_out, /*slot id*/ 0);

  //   // Set History for output set current Grad Node for
  //   egr::EagerUtils::SetHistory(p_autograd_out, run_program_node);
  // }
}
