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
#include "paddle/fluid/inference/analysis/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class PassManager;

/*
 * PassManagerMain - Executes all the PassManagers.
 */
class PassManagerMain {
 public:
  // Execute all the PassManagers registered.
  static void RunAll();

  // Register a pass manager with its name.
  static void Register(const std::string &name,
                       std::unique_ptr<PassManager> &&obj);

  // Get a pass manager called name, return nullptr if not exists.
  static PassManager *Lookup(const std::string &name);

 private:
  PADDLE_DISALLOW_COPY_AND_ASSIGN(PassManagerMain)
  static std::unordered_map<std::string, std::unique_ptr<PassManager>> data_;
};

/*
 * PassManager is the base class for all pass managers, a pass manager has
 * several Pass-es registered, and execute them in the right order.
 */
class PassManager {
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

  // Call all the passes' Initialize methods.
  virtual bool Initialize() = 0;

  // Run all the passes.
  virtual void RunAll() = 0;

  // Call all the passes' Finalize methods.
  virtual bool Finalize() = 0;

  // Register a PassT into this pass manager.
  template <typename PassT>
  virtual PassT *Register(const std::string &name) {}

 protected:
  Type type_;
};

// A pass manager that traverse the graph in DFS order.
template <typename GraphType>
class DFSPassManager : public PassManager {
 public:
  DFSPassManager(const GraphType &graph);

  bool Initialize() override;
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
class CustomIterPassManager : public PassManager {
 public:
  CustomIterPassManager() : type_(kCustomIter) {}
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
