// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <mutex>
#include <string>

#include "paddle/cinn/ir/schedule/ir_schedule.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/utils/registry.h"
#include "paddle/utils/flags.h"

namespace cinn {

namespace auto_schedule {

struct InitialTaskInfo {
  std::string task_key;
  ir::ModuleExpr module_expr;

  InitialTaskInfo(const std::string& task_key,
                  const ir::ModuleExpr& module_expr)
      : task_key(task_key), module_expr(module_expr) {}
};

// Global task registry, used to save the initial ModuleExpr of each task.
class InitialTaskRegistry : public Registry<InitialTaskInfo> {
 public:
  static InitialTaskRegistry* Global() {
    static InitialTaskRegistry x;
    return &x;
  }

  // Get the initial ModuleExpr of a task.
  inline const InitialTaskInfo* Get(const std::string& task_key) {
    const InitialTaskInfo* task_info =
        Registry<InitialTaskInfo>::Find(task_key);
    CHECK(task_info) << "InitialTaskInfo [" << task_key
                     << "] is not registered";
    return task_info;
  }

  // Check if the task info with task_key exists;
  inline const bool Has(const std::string& task_key) {
    return nullptr != Registry<InitialTaskInfo>::Find(task_key);
  }

  // Regist the initial ModuleExpr of a task into the map
  inline void Regist(const std::string& task_key,
                     const ir::ModuleExpr& module_expr) {
    std::lock_guard<std::mutex> guard(registering_mutex);
    if (fmap_.count(task_key) == 0) {
      InitialTaskInfo* task_info =
          new InitialTaskInfo(task_key, ir::ir_utils::IRCopy(module_expr));
      __REGISTER__(task_key, task_info);
    }
  }

 private:
  InitialTaskRegistry() = default;
  CINN_DISALLOW_COPY_AND_ASSIGN(InitialTaskRegistry);

  // Regist the initial ModuleExpr of a task.
  inline InitialTaskInfo* __REGISTER__(const std::string& task_key,
                                       InitialTaskInfo* task_info) {
    fmap_[task_key] = task_info;
    const_list_.push_back(task_info);
    entry_list_.push_back(task_info);
    return task_info;
  }
};

}  // namespace auto_schedule
}  // namespace cinn
