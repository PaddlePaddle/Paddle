/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class PianoJitOpSearchAndReplacePass : public Pass {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

class PianoJitOpSearchHelper {
 private:
  // TODO(levi): need to fill jit_supported ops list.
  std::unordered_set<const std::string> jit_supported_op_names = {
  "add",
  "sub",
  "mul",
  "mod",
  "div",
  };

 public:
  bool IsJitSupported(const std::string& op_name) {
    return jit_supported_op_names.find(op_name) != jit_supported_op_names.end();
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
