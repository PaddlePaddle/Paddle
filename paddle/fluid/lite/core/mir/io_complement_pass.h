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

#pragma once

#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

static void UpdateInputTo(framework::proto::OpDesc* desc,
                          const std::string& from, const std::string& to) {
  for (auto& item : *desc->mutable_inputs()) {
    for (auto& input : *item.mutable_arguments()) {
      if (input == from) {
        LOG(INFO) << "** update input argument from " << from << " to " << to;
        input = to;
      }
    }
  }
}

/*
 * IoComplementPass complement the necessary instruction to make data
 * transferring or transformation between different places.
 */
class IoComplementPass : public ProgramPass {
 public:
  void Apply(std::unique_ptr<mir::SSAGraph>& graph) override;

  void ComplementInputs(SSAGraph* graph, Node* inst_node, Node* in);

  void AddIoCopyInst(const Type& from, const Type& to, const std::string& var,
                     SSAGraph* graph, Node* inst_node,
                     const std::vector<Place>& valid_places);

  void SetValidPlaces(const std::vector<Place>& valid_places);

  // Pick the right kernel of IoCopy considering the input and output Type.
  void PickIoCopyKernel(SSAGraph* graph);

  const std::vector<Place>& valid_places() const { return valid_places_; };

 private:
  std::vector<Place> valid_places_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
