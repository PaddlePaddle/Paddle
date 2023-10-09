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
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"

namespace pir {
class Operation;
}

namespace paddle {
namespace framework {
class Scope;

class CinnJitInstruction : public InstructionBase {
 public:
  CinnJitInstruction(size_t id,
                     const platform::Place& place,
                     ::pir::Operation* op,
                     Scope* scope);

  // TODO(Aurelius84): Only implement core interface and need implement GC and
  // Event logic.
  void Run() override;

  const std::string& Name() const override;

  ::pir::Operation* Operation() const override { return op_; }

 private:
  class Impl;
  std::shared_ptr<Impl> impl_{nullptr};

  ::pir::Operation* op_{nullptr};  // not owned
};

}  // namespace framework
}  // namespace paddle
