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
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * AnalysisPass is a pass used to control the IR passes.
 */
class AnalysisPass {
 public:
  AnalysisPass() = default;
  virtual ~AnalysisPass() = default;

  // User should implement these.
  virtual bool InitializeImpl() = 0;
  virtual bool FinalizeImpl() = 0;
  virtual void RunImpl() = 0;

  // Mutable Pass.
  virtual bool Initialize(Argument *argument) {
    argument_ = argument;
    return true;
  }

  // Virtual method overriden by subclasses to do any necessary clean up after
  // all passes have run.
  virtual bool Finalize() { return true; }

  // Run on a single Graph.
  virtual void Run() = 0;

  // Human-readable short representation.
  virtual std::string repr() const = 0;
  // Human-readable long description.
  virtual std::string description() const { return "No DOC"; }

 protected:
  Argument *argument_{nullptr};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
