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
 * This file defines the interface for pass management.
 */

#pragma once

#include <string>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class PassManager;

/*
 * PassManagerMain - Executes all the PassManagers.
 */
class PassManagerMain : public OrderedRegistry<PassManager> {
 public:
  static PassManagerMain &Global() {
    static auto *x = new PassManagerMain;
    return *x;
  }

  // Execute all the PassManagers registered.
  void RunAll(const framework::proto::ProgramDesc &desc);

  PADDLE_DISALLOW_COPY_AND_ASSIGN(PassManagerMain)

 protected:
  DataFlowGraph data_flow_graph_;

 private:
  PassManagerMain() = default;
};

/*
 * PassManager is the base class for all pass managers, a pass manager has
 * several Pass-es registered, and execute them in the right order.
 */
class PassManager : public OrderedRegistry<Pass> {
 public:
  enum Type {
    kUnknown = -1,
    // The outer iteration is DFS algorithm.
    kDFS_PM,
    // The outer iteratoin is BFS algorithm.
    kBFS_PM,
    // The outer iteration follows a customized order.
    kCustomIter
  };

  // Call all the passes' Initialize methods. The desc and data_flow_graph are
  // globally shared, so pass them as the arguemnts for all the pass managers.
  virtual bool Initialize(const framework::proto::ProgramDesc &desc,
                          DataFlowGraph *data_flow_graph) = 0;

  // Run all the passes.
  virtual void RunAll() = 0;

  // Call all the passes' Finalize methods.
  virtual bool Finalize() = 0;

  virtual ~PassManager() {}

 protected:
  Type type_{Type::kUnknown};
};

// A pass manager that traverse the graph in DFS order.
template <typename GraphType>
class DFSPassManager : public PassManager {
 public:
  DFSPassManager(const GraphType &graph);

  bool Initialize(const framework::proto::ProgramDesc &desc,
                  DataFlowGraph *data_flow_graph) override;
  bool Finalize() override;
  // DFS traverse the graph, call the passes in each step.
  void RunAll() override;

 private:
  GraphType graph_;
};

// TODO(Superjomn) Implement BFSPassManager if needed.

/*
 * A pass manager that traverse the graph in a customized order, it is a virtual
 * class and need to be override by sub-classes.
 */
class DFG_PassManager : public PassManager {
 public:
  DFG_PassManager();
  bool Initialize(const framework::proto::ProgramDesc &desc,
                  DataFlowGraph *data_flow_graph) override {
    graph_ = data_flow_graph;
    for (auto &pass : data_) {
      pass->Initialize();
      pass->Initialize(desc);
    }
    return true;
  }

  void RunAll() override;

  bool Finalize() override {
    for (auto &pass : data_) {
      pass->Finalize();
    }
    return true;
  }

 private:
  DataFlowGraph *graph_;
};


// Run all the pass managers to analysis and optimize the graph.
static void RunAnalysis() {}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
