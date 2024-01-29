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

  void SetBwdPrimEnabled(bool enable_prim) { enable_bwd_prim_ = enable_prim; }

  bool IsBwdPrimEnabled() { return enable_bwd_prim_; }

  void SetFwdPrimEnabled(bool enable_prim) { enable_fwd_prim_ = enable_prim; }

  bool IsFwdPrimEnabled() { return enable_fwd_prim_; }

  void SetEagerPrimEnabled(bool enable_prim) {
    enable_eager_prim_ = enable_prim;
  }

  bool IsEagerPrimEnabled() { return enable_eager_prim_; }

  void SetAllPrimEnabled(bool enable_prim) {
    enable_fwd_prim_ = enable_prim;
    enable_bwd_prim_ = enable_prim;
  }

  size_t CheckSkipCompOps(const std::string& op_type) const {
    return skip_comp_ops_.count(op_type);
  }

  void AddSkipCompOps(const std::string& op_type) {
    skip_comp_ops_.insert(op_type);
  }

  void RemoveSkipCompOps(const std::string& op_type) {
    skip_comp_ops_.erase(op_type);
  }

  void SetTargetGradName(const std::map<std::string, std::string>& m) {
    target_grad_name_ = m;
  }

  std::map<std::string, std::string> GetTargetGradName() {
    return target_grad_name_;
  }

 private:
  StaticCompositeContext()
      : current_block_desc_(nullptr),
        generator_(new UniqueNameGenerator()),
        skip_comp_ops_({"matmul_v2"}) {}
  // TODO(Ruting) test cases when fix static backward
  framework::BlockDesc* current_block_desc_;
  std::unique_ptr<UniqueNameGenerator> generator_;
  std::unordered_set<std::string> skip_comp_ops_;
  std::map<std::string, std::string> target_grad_name_;
  static thread_local bool enable_bwd_prim_;
  static thread_local bool enable_fwd_prim_;
  static thread_local bool enable_eager_prim_;
  TEST_API static StaticCompositeContext* static_composite_context_;
  DISABLE_COPY_AND_ASSIGN(StaticCompositeContext);
};

}  // namespace prim
}  // namespace paddle
