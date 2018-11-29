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

#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class AsyncExecutor {
 public:
  AsyncExecutor(Scope* scope, const platform::Place& place);
  virtual ~AsyncExecutor() {}
  void RunFromFile(const ProgramDesc& main_program,
                   const std::string& data_feed_desc_str,
                   const std::vector<std::string>& filelist,
                   const int thread_num,
                   const std::vector<std::string>& fetch_names,
                   const bool debug = false);

 private:
  void CreateThreads(ExecutorThreadWorker* worker,
                     const ProgramDesc& main_program,
                     const std::shared_ptr<DataFeed>& reader,
                     const std::vector<std::string>& fetch_var_names,
                     Scope* root_scope, const int thread_index,
                     const bool debug);

 public:
  Scope* root_scope_;
  platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
