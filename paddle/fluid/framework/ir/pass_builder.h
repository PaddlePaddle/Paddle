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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Pass;

class PassBuilder {
 public:
  PassBuilder() {}

  virtual ~PassBuilder() {}

  // Append a new pass to the end.
  std::shared_ptr<Pass> AppendPass(const std::string& pass_type);

  // Insert a new pass after `idx`.
  std::shared_ptr<Pass> InsertPass(size_t idx, const std::string& pass_type);

  // Remove a new pass at `idx`.
  void RemovePass(size_t idx);

  // Returns a list of all passes.
  std::vector<std::shared_ptr<Pass>> AllPasses() const { return passes_; }

 protected:
  std::vector<std::shared_ptr<Pass>> passes_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
