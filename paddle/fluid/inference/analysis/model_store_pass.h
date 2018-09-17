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

/*
 * This file defines ModelStorePass, which store the runtime DFG to a Paddle
 * model in the disk, and that model can be reloaded for prediction.
 */

#pragma once
#include <string>
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class ModelStorePass : public DataFlowGraphPass {
 public:
  bool Initialize(Argument* argument) override {
    if (!argument) {
      LOG(ERROR) << "invalid argument";
      return false;
    }
    argument_ = argument;
    return true;
  }

  void Run(DataFlowGraph* x) override;

  std::string repr() const override { return "DFG-store-pass"; }
  std::string description() const override {
    return R"DD(This file defines ModelStorePass, which store the runtime DFG to a Paddle
    model in the disk, and that model can be reloaded for prediction again.)DD";
  }

  bool Finalize() override;

 private:
  Argument* argument_{nullptr};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
