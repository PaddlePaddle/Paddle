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

class Pass {
 public:
  Pass() = default;
  virtual ~Pass() = default;
  // Mutable Pass.
  virtual bool Initialize(Argument *argument) { return false; }
  // Readonly Pass.
  virtual bool Initialize(const Argument &argument) { return false; }

  // Virtual method overriden by subclasses to do any necessary clean up after
  // all passes have run.
  virtual bool Finalize() { return false; }

  // Get a Pass appropriate to print the Node this pass operates on.
  virtual Pass *CreatePrinterPass(std::ostream &os,
                                  const std::string &banner) const {
    return nullptr;
  }

  // Create a debugger Pass that draw the DFG by graphviz toolkit.
  virtual Pass *CreateGraphvizDebugerPass() const { return nullptr; }

  // Run on a single Node.
  virtual void Run(Node *x) { LOG(FATAL) << "not valid"; }
  // Run on a single Function.
  virtual void Run(Function *x) { LOG(FATAL) << "not valid"; }
  // Run on a single FunctionBlock.
  virtual void Run(FunctionBlock *x) { LOG(FATAL) << "not valid"; }
  // Run on a single DataFlowGraph.
  virtual void Run(DataFlowGraph *x) { LOG(FATAL) << "not valid"; }

  // Human-readable short representation.
  virtual std::string repr() const = 0;
  // Human-readable long description.
  virtual std::string description() const = 0;
};

// NodePass process on any Node types.
class NodePass : public Pass {
 public:
  virtual void Run(Node *node) = 0;
};

// NodePass process on any Function node types.
class FunctionPass : public Pass {
 public:
  virtual void Run(Function *node) = 0;
};

// NodePass process on any FunctionBlock node types.
class FunctionBlockPass : public Pass {
 public:
  virtual void Run(FunctionBlock *node) = 0;
};

// GraphPass processes on any GraphType.
class DataFlowGraphPass : public Pass {
 public:
  virtual void Run(DataFlowGraph *graph) = 0;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
