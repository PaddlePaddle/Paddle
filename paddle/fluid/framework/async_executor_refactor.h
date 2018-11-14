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

#ifndef PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
#define PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_

#include <memory>
#include <mutex>    // NOLINT
#include <set>
#include <map>
#include <string>
#include <thread>   // NOLINT
#include <vector>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/datafeed_creator.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

class AsyncExecutor {
 public:
  explicit AsyncExecutor(const platform::Place& place);
  virtual ~AsyncExecutor() {}
  void SetRootScope(const Scope* root_scope);
  void CheckFiles(
      const std::vector<std::string>& files);
  void RunFromFiles(
      const ProgramDesc& main_program,
      const DataFeedDesc& data_feed_desc,
      const std::vector<std::string>& files,
      const int thread_num);

 public:
  Scope* root_scope_;
  platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
#endif  // PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
