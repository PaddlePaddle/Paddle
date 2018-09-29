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

#pragma once

#include <glog/logging.h>
#include <iosfwd>
#include <string>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/data_flow_graph.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/node.h"

namespace paddle {
namespace inference {
namespace analysis {

class AnalysisPass {
 public:
  AnalysisPass() = default;
  virtual ~AnalysisPass() = default;
  // Mutable Pass.
  virtual bool Initialize(Argument *argument) { return false; }
  // Readonly Pass.
  virtual bool Initialize(const Argument &argument) { return false; }

  // Virtual method overriden by subclasses to do any necessary clean up after
  // all passes have run.
  virtual bool Finalize() { return false; }

  // Create a debugger Pass that draw the DFG by graphviz toolkit.
  virtual AnalysisPass *CreateGraphvizDebugerPass() const { return nullptr; }

  // Run on a single DataFlowGraph.
  virtual void Run(DataFlowGraph *x) = 0;

  // Human-readable short representation.
  virtual std::string repr() const = 0;
  // Human-readable long description.
  virtual std::string description() const { return "No DOC"; }
};

// GraphPass processes on any GraphType.
class DataFlowGraphPass : public AnalysisPass {};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
