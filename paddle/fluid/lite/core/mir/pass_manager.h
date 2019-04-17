// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <list>
#include <map>
#include <string>

#include "paddle/fluid/lite/core/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class PassManager {
 public:
  static PassManager& Global() {
    static PassManager x;
    return x;
  }

  PassManager();

  void Run(std::unique_ptr<SSAGraph>& graph) {
    for (auto& pass : passes_) {
      LOG(INFO) << "Running MIR pass " << pass->name();
      pass->Apply(graph);
    }
  }

  bool AddNewPass(const std::string& name, Pass* pass) {
    passes_.emplace_back(pass);
    pass_map_.emplace(name, passes_.back().get());
    passes_.back()->set_name(name);
    return true;
  }

  // Clear all the passes.
  void Clear() { passes_.clear(); }

  std::list<std::unique_ptr<mir::Pass>>::iterator passes_begin() {
    return passes_.begin();
  }
  std::list<std::unique_ptr<mir::Pass>>::iterator passes_end() {
    return passes_.end();
  }
  std::list<std::unique_ptr<mir::Pass>>::const_iterator passes_const_begin()
      const {
    return passes_.begin();
  }
  std::list<std::unique_ptr<mir::Pass>>::const_iterator passes_const_end()
      const {
    return passes_.end();
  }

  Pass* LookUp(const std::string& key) {
    auto it = pass_map_.find(key);
    if (it != pass_map_.end()) return it->second;
    return nullptr;
  }

  template <typename PassTy>
  PassTy* LookUp(const std::string& key) {
    auto it = pass_map_.find(key);
    if (it != pass_map_.end()) return dynamic_cast<PassTy*>(it->second);
    return nullptr;
  }

 private:
  std::list<std::unique_ptr<mir::Pass>> passes_;
  std::map<std::string, mir::Pass*> pass_map_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
