// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"

namespace paddle {
namespace prim {
class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

class StaticCompositeContext {
 public:
  static StaticCompositeContext& Instance() {
    return *static_composite_context_;
  }

  framework::BlockDesc* GetBlock() { return current_block_desc_; }

  void SetBlock(framework::BlockDesc* new_block) {
    current_block_desc_ = new_block;
  }

  std::string GenerateUniqueName(std::string key = "composite_tmp") {
    return generator_->Generate(key);
  }

  void SetPrimEnabled(bool enable_prim) { enable_prim_ = enable_prim; }

  bool IsPrimEnabled() { return enable_prim_; }

 private:
  StaticCompositeContext()
      : current_block_desc_(nullptr), generator_(new UniqueNameGenerator()) {}

  framework::BlockDesc* current_block_desc_;
  std::unique_ptr<UniqueNameGenerator> generator_;
  static thread_local bool enable_prim_;
  static StaticCompositeContext* static_composite_context_;
  DISABLE_COPY_AND_ASSIGN(StaticCompositeContext);
};

}  // namespace prim
}  // namespace paddle
