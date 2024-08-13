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

#include "paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.h"

namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class ValueExecutionInfo;

class OneDNNMixedPhiKernelInstruction : public OneDNNPhiKernelInstruction {
 public:
  OneDNNMixedPhiKernelInstruction(size_t id,
                                  const phi::Place& place,
                                  ::pir::Operation* op,
                                  const ValueExecutionInfo* value_exec_info);

  void Run() override;

 private:
  std::string kernel_name_;
  phi::KernelKey kernel_key_;
  bool has_choose_kernel_{false};
  bool use_onednn_kernel_{true};
};

}  // namespace framework
}  // namespace paddle
