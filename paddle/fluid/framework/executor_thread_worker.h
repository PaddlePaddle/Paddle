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
#include <vector>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
void CreateTensor(Variable* var, proto::VarType::Type var_type);

class ExecutorThreadWorker {
 public:
  ExecutorThreadWorker()
      : thread_id_(-1), root_scope_(NULL), thread_scope_(NULL), debug_(false) {}
  ~ExecutorThreadWorker() {}

  void CreateThreadResource(const framework::ProgramDesc& program,
                            const paddle::platform::Place& place);
  void SetThreadId(int tid);
  void SetDebug(const bool debug) { debug_ = debug; }
  void SetRootScope(Scope* g_scope);
  // set cpu device in this function
  // cpu binding is used by default
  void SetDevice();
  // since we read data into memory that can not be accessed by program
  // we need to bind memory of data with corresponding variables in program
  // this function should be called after data feed is set
  void BindingDataFeedMemory();
  // set data feed declared in executor
  void SetDataFeed(const std::shared_ptr<DataFeed>& datafeed);
  // A multi-thread training function
  void TrainFiles();
  // set fetch variable names from python interface assigned by users
  void SetFetchVarNames(const std::vector<std::string>& fetch_var_names);

 private:
  void CreateThreadScope(const framework::ProgramDesc& program);
  void CreateThreadOperators(const framework::ProgramDesc& program);
  void SetMainProgram(const ProgramDesc& main_program_desc);
  void SetPlace(const paddle::platform::Place& place);

 protected:
  // thread index
  std::shared_ptr<DataFeed> thread_reader_;  // shared queue, thread buffer
  int thread_id_;
  // operator name
  std::vector<std::string> op_names_;
  // thread level, local operators for forward and backward
  std::vector<OperatorBase*> ops_;
  // main program for training
  std::unique_ptr<framework::ProgramDesc> main_program_;
  // execution place
  platform::Place place_;
  // root scope for model parameters
  Scope* root_scope_;
  // a thread scope, father scope is global score which is shared
  Scope* thread_scope_;

 private:
  std::vector<std::string> fetch_var_names_;
  std::vector<std::vector<float>> fetch_values_;
  bool debug_;
};

}  // namespace framework
}  // namespace paddle
