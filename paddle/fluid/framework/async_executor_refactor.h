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
void CreateTensor(Variable* var, proto::VarType::Type var_type);

class ExecutorThreadWorker {
 public:
  ExecutorThreadWorker() {}
  ~ExecutorThreadWorker() {}
  void CreateThreadScope(const framework::ProgramDesc& program);
  void SetDataFeed(const DataFeed& datafeed);
  void SetThreadId(int tid);
  void CreateThreadOperators(const framework::ProgramDesc& program);
  void SetRootScope(Scope* g_scope);
  void SetDevice();
  void SetMainProgram(const ProgramDesc& main_program_desc);
  void SetPlace(const paddle::platform::Place& place);
  void BindingDataFeedMemory();
  void SetSparseCommData(const std::map<std::string, int>& param_names);
  void SetDataFeed(const std::shared_ptr<DataFeed>& datafeed);

 protected:
  // thread index
  std::shared_ptr<DataFeed> thread_reader_;  // shared queue, thread buffer
  int thread_id_;
  // op name
  std::vector<std::string> op_names_;
  // local ops for forward and backward
  std::vector<OperatorBase *> ops_;
  // main program for training
  std::unique_ptr<framework::ProgramDesc> main_program_;
  // execution place
  platform::Place place_;
  // root scope for model parameters
  Scope* root_scope_;
  // a thread scope, father scope is global score which is shared
  Scope* thread_scope_;
};

class AsyncExecutor {
 public:
  explicit AsyncExecutor(const platform::Place& place);
  virtual ~AsyncExecutor() {}
  void SetRootScope(const Scope* root_scope);
  Scope* GetRootScope() { return root_scope_; }
  void CheckFiles(const std::vector<std::string>& files);
  void RunFromFiles(
      const ProgramDesc& main_program,
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
