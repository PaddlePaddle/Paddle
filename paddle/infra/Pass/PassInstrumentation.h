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

/// The code and design is mainly from mlir, very thanks to the great project.

#pragma once

#include <memory>
#include <optional>
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
namespace infra {
class Pass;

namespace detail {
struct PassInstrumentorImpl;
}  // namespace detail

class PassInstrumentation {
 public:
  virtual ~PassInstrumentation() = 0;

  /// A callback to run before a pass pipeline is executed.
  virtual void RunBeforePipeline(std::optional<mlir::OperationName> name);

  virtual void RunAfterPipeline(std::optional<mlir::OperationName> name);

  virtual void RunBeforePass(Pass* pass, mlir::Operation* op);

  virtual void RunAfterPass(Pass* pass, mlir::Operation* op);

  virtual void RunBeforeAnalysis(const std::string& name,
                                 mlir::TypeID id,
                                 mlir::Operation* op);

  virtual void RunAfterAnalysis(const std::string& name,
                                mlir::TypeID id,
                                mlir::Operation* op);
};

/// This class holds a collection of PassInstrumentation obejcts, and invokes
/// their respective callbacks.
class PassInstrumentor {
 public:
  PassInstrumentor();
  ~PassInstrumentor();
  PassInstrumentor(PassInstrumentor&&) = delete;
  PassInstrumentor(const PassInstrumentor&) = delete;

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

  void RunBeforePipeline(std::optional<mlir::OperationName> name);

  void RunAfterPipeline(std::optional<mlir::OperationName> name);

  void RunBeforePass(Pass* pass, mlir::Operation* op);

  void RunAfterPass(Pass* pass, mlir::Operation* op);

  void RunBeforeAnalysis(const std::string& name,
                         mlir::TypeID id,
                         mlir::Operation* op);

  void RunAfterAnalysis(const std::string& name,
                        mlir::TypeID id,
                        mlir::Operation* op);

  // TODO(wilber): Other hooks.

 private:
  std::unique_ptr<detail::PassInstrumentorImpl> impl;
};

}  // namespace infra
