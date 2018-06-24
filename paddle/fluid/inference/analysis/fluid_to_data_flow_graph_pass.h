/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

/*
 * This file implements the transformation from data flow graph to fluid
 * ProgramDesc.
 */

#pragma once

#include <string>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Transform a FluidDesc to a data flow graph.
 */
class FluidToDataFlowGraphPass final : public DataFlowGraphPass {
 public:
  FluidToDataFlowGraphPass() = default;

  bool Initialize(Argument *argument) override;
  bool Finalize() override;

  void Run(DataFlowGraph *graph) override;

  std::string repr() const override { return "fluid-to-data-flow-graph"; }
  std::string description() const override {
    return "transform a fluid ProgramDesc to a data flow graph.";
  }

  Pass *CreatePrinterPass(std::ostream &os,
                          const std::string &banner) const override;

 private:
  framework::proto::ProgramDesc const *desc_;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
