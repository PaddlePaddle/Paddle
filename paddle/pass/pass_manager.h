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
#include <vector>

namespace ir {

class IrContext;
class Operation;
class Pass;

namespace detail {
class PassAdaptor;
}

class PassManager {
 public:
  explicit PassManager(ir::IrContext *context, uint8_t opt_level = 2);

  ~PassManager() = default;

  const std::vector<std::unique_ptr<Pass>> &GetPasses() const {
    return passes_;
  }

  bool Empty() const { return passes_.empty(); }

  ir::IrContext *GetContext() const { return context_; }

  bool Run(ir::Operation *op);

  void AddPass(std::unique_ptr<Pass> pass) {
    passes_.emplace_back(std::move(pass));
  }

 private:
  bool RunPasses(ir::Operation *op);

  bool Initialize(ir::IrContext *context);

 private:
  ir::IrContext *context_;

  uint8_t opt_level_;

  std::vector<std::unique_ptr<Pass>> passes_;

  std::unique_ptr<Pass> pass_adaptor_;

  friend class detail::PassAdaptor;
};

}  // namespace ir
