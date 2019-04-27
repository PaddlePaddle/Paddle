// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/core/optimizer.h"
#include "paddle/fluid/lite/core/mir/io_complement_pass.h"
#include "paddle/fluid/lite/core/mir/static_kernel_pick_pass.h"

namespace paddle {
namespace lite {

void Optimizer::SpecifyKernelPickTactic(core::KernelPickFactor factor) {
  auto* pass = mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
      "static_kernel_pick_pass");
  CHECK(pass);

  *pass->mutable_kernel_pick_factors() = factor;
}

void Optimizer::RunPasses() {
  std::vector<std::string> passes({
      "static_kernel_pick_pass",        //
      "variable_place_inference_pass",  //
      "argument_type_display_pass",     //
      "io_complement_pass",             //
      "argument_type_display_pass",     //
      "variable_place_inference_pass",  //
      "argument_type_display_pass",     //
      "io_copy_kernel_pick_pass",       //
      "variable_place_inference_pass",  //
      "runtime_context_assign_pass",    //
  });
  for (auto& pass_type : passes) {
    LOG(INFO) << ".. running pass " << pass_type;
    auto* pass = mir::PassManager::Global().LookUp(pass_type);
    CHECK(pass);
    if (pass->name() == "io_complement_pass") {
      auto* _pass = dynamic_cast<mir::IoComplementPass*>(pass);
      _pass->SetValidPlaces(valid_places_);
      CHECK(!_pass->valid_places().empty());
      _pass->Apply(graph_);
    } else {
      pass->Apply(graph_);
    }
  }
  // mir::PassManager::Global().Run(graph_);
}
}  // namespace lite
}  // namespace paddle
