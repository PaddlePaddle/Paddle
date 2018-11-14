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
  void CreateThreadScope(const ProgramDesc& program);
  void SetThreadId(int tid);
  void CreateThreadOperators(const ProgramDesc& program);
  void SetRootScope(Scope* g_scope);
  void SetDevice();
  void AddFidSet();
  void SetCommBatch(int comm_batch) { comm_batch_ = comm_batch; }
  void AddTrainFile(const std::string& filename);
  void SetMainProgram(const ProgramDesc& main_program_desc);
  void SetPlace(const paddle::platform::Place& place);
  void SetMaxTrainingEpoch(const int max_epoch);
  void BindingDataFeedMemory();

  void SetModelPrefix(const std::string& prefix) { model_prefix_ = prefix; }

  void SetInspectVarNames(const std::vector<std::string>& inspect_var_names);
  void SetModelParamNames(const std::vector<std::string>& param_names);
  void SetDataFeed(DataFeed& datafeed); // NOLINT
  void Train();
  const char* PickOneFile();
  void UpdateEpochNum();
  void Reset();

  void Initialize() {}
  std::vector<float>& GetInspectValues() {return inspect_values_;}

 protected:
  // thread index
  int thread_id_;

  // max epoch for each thread
  unsigned int max_epoch_;

  // instances learned currently
  int comm_batch_;
  std::string model_prefix_;
  std::vector<std::string> op_names_;

  // local ops for forward and backward
  std::vector<OperatorBase *> ops_;

  // main program for training
  std::unique_ptr<ProgramDesc> main_program_;

  // binary data reader
  std::unique_ptr<DataFeed> local_reader_;

  std::vector<std::string> inspect_var_names_;
  std::vector<std::string> model_param_names_;

  // execution place
  platform::Place place_;

  // root scope for model parameters
  Scope* root_scope_;

  // a thread scope, father scope is global score which is shared
  Scope* thread_scope_;

 private:
  std::vector<float> inspect_values_;
};

class AsyncExecutor {
 public:
  explicit AsyncExecutor(ProgramDesc& main_program,     // NOLINT
                         const std::vector<std::string>& param_names,
                         TextClassDataFeed& data_feed,  // NOLINT
                         unsigned int thread_num,
                         const platform::Place& place);
  virtual ~AsyncExecutor() {}
  static std::unique_ptr<ProgramDesc> LoadDescFromFile(
                                          const std::string& filename);
  void InitRootScope(Scope* scope);
  void SetMaxTrainingEpoch(const int max_epoch);
  Scope* GetRootScope() { return root_scope_; }
  void SetBatchSize(const int batch_size) { batch_size_ = batch_size; }

  void SetCommBatch(int comm_batch) {
    comm_batch_ = comm_batch;
  }

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
  virtual void PrepareThreads(const ProgramDesc& host_program);
  void RunStartupProgram(const ProgramDesc& program, Scope* scope);
  std::vector<float>& Run(const std::vector<std::string>& inspect_var_names);

  void LoadInitModel();

 private:
  void SetInspectVarNames(const std::vector<std::string>& inspect_var_names);

 public:
  int thread_num_;
  int max_epoch_;
  int batch_size_;
  int comm_batch_;
  std::vector<std::shared_ptr<ExecutorThreadWorker> > workers_;
  std::vector<std::thread> threads_;
  std::vector<std::string> inspect_var_names_;
  std::vector<std::string> model_param_names_;
  std::string model_prefix_;
  std::string model_path_;
  std::string init_prog_file_;
  std::string init_model_file_;
  Scope* root_scope_;
  platform::Place place_;

 private:
  ProgramDesc& main_program_;
  TextClassDataFeed& data_feed_;
  std::vector<float> inspect_values_;

 private:
  static bool workers_initialized_;
};

}  // namespace framework
}  // namespace paddle
#endif  // PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
