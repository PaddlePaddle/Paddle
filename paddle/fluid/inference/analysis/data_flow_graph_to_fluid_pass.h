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
 * This file implements the transformation from fluid ProgramDesc to data flow
 * graph.
 */

#pragma once

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {
class DataFlowGraphToFluidPass final : public DataFlowGraphPass {
 public:
  DataFlowGraphToFluidPass() = default;

  bool Initialize(Argument *argument) override;
  bool Finalize() override;

  void Run(DataFlowGraph *graph) override;

  std::string repr() const override { return "DFG to fluid"; }
  std::string description() const override {
    return "Transform a DFG to a Fluid ProgramDesc";
  }

  Pass *CreatePrinterPass(std::ostream &os,
                          const std::string &banner) const override {
    return nullptr;
  }

 protected:
  // Add a Fluid Op into the ProgramDesc.
  void AddFluidOp(Node *node);
  // Add a EngineOp into the ProgramDesc.
  void AddEngineOp(Node *node);

 private:
  framework::proto::ProgramDesc *desc_;
};
}  // namespace analysis
}  // namespace inference
}  // namespace paddle
