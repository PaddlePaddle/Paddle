/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/common/thread_pool.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"

namespace paddle {
namespace platform {
namespace gcu {

class GcuExecutorImpl;
class GcuExecutor {
 public:
  explicit GcuExecutor(const framework::Scope* scope);
  ~GcuExecutor() = default;
  GcuExecutor(const GcuExecutor& exec) = default;
  GcuExecutor& operator=(const GcuExecutor& exec) = default;
  void ReleaseResource();
  void ReleaseMemory();
  void ResetScope(const framework::Scope* scope);
  void RunGcuOp(const std::vector<const Tensor*>& inputs,
                const std::vector<Tensor*>& outputs,
                const paddle::framework::ExecutionContext& ctx,
                const std::string& program_key = "default_program_key",
                const int train_flag = -1,
                const framework::Scope* curr_scope = nullptr);
  void SetResouceAllocFlag(bool flag);
  void Synchronize();

 private:
  std::shared_ptr<GcuExecutorImpl> impl_ = nullptr;
};

class GcuExecutorManager {
 public:
  ~GcuExecutorManager() { ReleaseAll(); }
  void ReleaseAll() {
    for (const auto& p : pg_to_executor_) {
      p.second->ReleaseResource();
    }
    pg_to_executor_.clear();
  }
  void Add(const std::string& key, std::shared_ptr<GcuExecutor> exec) {
    pg_to_executor_[key] = exec;
  }
  std::shared_ptr<GcuExecutor> Find(const std::string& key) {
    if (pg_to_executor_.count(key) == 0) {
      return nullptr;
    }
    auto exec = pg_to_executor_[key];
    PADDLE_ENFORCE_NE(
        exec, nullptr, platform::errors::NotFound("buffered exec is nullptr"));
    return exec;
  }

 public:
  static GcuExecutorManager* GetInstance() {
    static GcuExecutorManager manager;
    return &manager;
  }

 private:
  std::map<std::string, std::shared_ptr<GcuExecutor>> pg_to_executor_;
};

void Synchronize(const framework::ProgramDesc& program);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
