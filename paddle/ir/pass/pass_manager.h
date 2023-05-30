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

#include <cstdint>
#include <memory>
#include <vector>

#include "paddle/ir/core/program.h"

namespace ir {

class IrContext;
class Operation;
class Program;
class Pass;
class PassInstrumentation;
class PassInstrumentor;

namespace detail {
class PassAdaptor;
}

class PassManager {
 public:
  explicit PassManager(ir::IrContext *context, uint8_t opt_level = 2);

  ~PassManager() = default;

  const std::vector<std::unique_ptr<Pass>> &passes() const { return passes_; }

  bool empty() const { return passes_.empty(); }

  ir::IrContext *context() const { return context_; }

  // bool Run(ir::Program *program) const;
  bool Run(ir::Operation *op) const;

  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

  void AddInstrumentation(std::unique_ptr<PassInstrumentation> pi);

 private:
  bool Initialize(ir::IrContext *context) const;

 private:
  ir::IrContext *context_;

  uint8_t opt_level_;

  bool verify_{true};

  std::vector<std::unique_ptr<Pass>> passes_;

  std::unique_ptr<Pass> pass_adaptor_;

  std::unique_ptr<PassInstrumentor> instrumentor_;

  friend class detail::PassAdaptor;
};

}  // namespace ir
