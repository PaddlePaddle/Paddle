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
#include <typeinfo>
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/datafeed_creator.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class AsyncExecutor {
 public:
  explicit AsyncExecutor(Scope& scope, const platform::Place& place);   // NOLINT
  virtual ~AsyncExecutor() {}
  static std::unique_ptr<ProgramDesc> LoadDescFromFile(
                                          const std::string& filename);
  Scope* GetRootScope() { return &root_scope_; }

  void SetModelPath(const std::string& model_path) {
    model_path_ = model_path;
  }

  void SetInitProgFile(const std::string& init_prog_file) {
    init_prog_file_ = init_prog_file;
  }

  void SetInitModelFile(const std::string& init_model_file) {
    init_model_file_ = init_model_file;
  }

  void SetModelPrefix(const std::string& model_prefix);
  void RunStartupProgram(const ProgramDesc& program, Scope* scope);
  std::vector<float> RunFromFile(const ProgramDesc& main_program,
                                  const DataFeedDesc& data_feed_desc,
                                  const std::vector<std::string>& filelist,
                                  const int thread_num,
                                  const std::vector<std::string>& fetch_names);

  void CheckFiles(const std::vector<std::string>& files);
  void LoadInitModel();

 private:
  void CreateThreads(ExecutorThreadWorker* worker,
                     const ProgramDesc& main_program,
                     const std::shared_ptr<DataFeed>& reader,
                     const std::vector<std::string>& fetch_var_names,
                     Scope& root_scope,   // NOLINT
                     const int thread_index);


 public:
  std::string model_prefix_;
  std::string model_path_;
  std::string init_prog_file_;
  std::string init_model_file_;
  Scope& root_scope_;
  platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
#endif  // PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
