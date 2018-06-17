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
 * This file defines the logic of pass management. The analysis for inference is
 * a pipeline of Passes, a PassManager is a agency that helps to manage the
 * executation of the Passes.
 *
 * There are two modes of Passes, the first one is called NodePass and takes
 * an Node as input and output; the second one is called DFGPass and takes a
 * DFG(Data Flow Graph) as input and output. It is hard to put all the passes in
 * the same pipeline, there are two kinds of PassManagers, both takes a DFG as
 * input and output a DFG, but the Passes inside are different:
 *
 *   1. NodePassManager: the passes inside are all NodePasses, it can have
 *      different graph trivial algorithm, for example, DFS_NodePassManager will
 *      trigger the passes in depth first order;
 *   2. DfgPassManager: the passes inside are all DfgPasses.
 */

#pragma once

#include <string>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * PassManager is the base class for all pass managers, a pass manager has
 * several Pass-es registered, and execute them in the linear order.
 */
class PassManager : public OrderedRegistry<Pass> {
 public:
  PassManager() = default;
  // Call all the passes' Initialize methods. The desc and data_flow_graph are
  // globally shared, so pass them as the arguemnts for all the pass managers.
  virtual bool Initialize(const Argument& argument) { return false; }

  virtual bool Initialize(Argument* argument) {
    argument_ = argument;
    for (auto& pass : data_) {
      LOG(INFO) << "Initializing pass " << pass->repr();
      if (!pass->Initialize(argument)) {
        LOG(ERROR) << "Failed to initialize pass [" << pass->repr() << "]";
        return false;
      }
    }
    return true;
  }

  // Call all the passes' Finalize methods.
  virtual bool Finalize() {
    for (auto& pass : data_) {
      if (!pass->Finalize()) {
        LOG(ERROR) << "Failed to finalize pass [" << pass->repr() << "]";
        return false;
      }
    }
    return true;
  }

  // Run all the passes.
  virtual void RunAll() = 0;

  // Short identifier.
  virtual std::string repr() const = 0;
  // Long description.
  virtual std::string description() const = 0;

  virtual ~PassManager() = default;

 protected:
  Argument* argument_{nullptr};
};

/*
 * A pass manager that process a DFG.
 */
class DfgPassManager : public PassManager {
 public:
  DfgPassManager() = default;

  void RunAll() override;

  virtual ~DfgPassManager() = default;
};

/*
 * A pass manager that process a Node each time.
 */
class NodePassManager : public PassManager {
 public:
  NodePassManager() = default;

  void RunAll() override;

  virtual ~NodePassManager() = default;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
