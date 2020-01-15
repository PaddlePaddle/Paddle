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

#include "paddle/fluid/framework/ir/pass_builder.h"
#include <memory>
#include <utility>

namespace paddle {
namespace framework {
namespace ir {

std::shared_ptr<Pass> PassBuilder::AppendPass(const std::string& pass_type) {
  VLOG(1) << "Append " << pass_type;
  auto pass = ir::PassRegistry::Instance().Get(pass_type);
  passes_.emplace_back(pass.release());
  return passes_.back();
}

void PassBuilder::RemovePass(size_t idx) {
  PADDLE_ENFORCE(passes_.size() > idx);
  passes_.erase(passes_.begin() + idx);
}

std::shared_ptr<Pass> PassBuilder::InsertPass(size_t idx,
                                              const std::string& pass_type) {
  PADDLE_ENFORCE(passes_.size() >= idx);
  std::shared_ptr<Pass> pass(
      ir::PassRegistry::Instance().Get(pass_type).release());
  passes_.insert(passes_.begin() + idx, std::move(pass));
  return passes_[idx];
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
