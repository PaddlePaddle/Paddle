// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>

#include "paddle/pir/include/core/type_id.h"

namespace pir {

class Operation;
class Pass;

namespace detail {
struct PassInstrumentorImpl;
}  // namespace detail

class PassInstrumentation {
 public:
  PassInstrumentation() = default;
  virtual ~PassInstrumentation() = default;

  // A callback to run before a pass pipeline is executed.
  virtual void RunBeforePipeline(Operation* op) {}

  // A callback to run after a pass pipeline is executed.
  virtual void RunAfterPipeline(Operation* op) {}

  // A callback to run before a pass is executed.
  virtual void RunBeforePass(Pass* pass, Operation* op) {}

  // A callback to run after a pass is executed.
  virtual void RunAfterPass(Pass* pass, Operation* op) {}

  // A callback to run before a analysis is executed.
  virtual void RunBeforeAnalysis(const std::string& name,
                                 TypeId id,
                                 Operation* op) {}

  // A callback to run after a analysis is executed.
  virtual void RunAfterAnalysis(const std::string& name,
                                TypeId id,
                                Operation* op) {}
};

/// This class holds a collection of PassInstrumentation objects, and invokes
/// their respective callbacks.
class IR_API PassInstrumentor {
 public:
  PassInstrumentor();
  ~PassInstrumentor();
  PassInstrumentor(PassInstrumentor&&) = delete;
  PassInstrumentor(const PassInstrumentor&) = delete;

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

  void RunBeforePipeline(Operation* op);

  void RunAfterPipeline(Operation* op);

  void RunBeforePass(Pass* pass, Operation* op);

  void RunAfterPass(Pass* pass, Operation* op);

  void RunBeforeAnalysis(const std::string& name, TypeId id, Operation* op);

  void RunAfterAnalysis(const std::string& name, TypeId id, Operation* op);

  // TODO(liuyuanle): Add other hooks.

 private:
  std::unique_ptr<detail::PassInstrumentorImpl> impl_;
};

}  // namespace pir
