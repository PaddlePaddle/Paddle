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
  virtual ~ExecutorThreadWorker() {}
  void CreateThreadScope(const framework::ProgramDesc& program);
  void SetDataFeed(const DataFeed& datafeed);
  void SetThreadId(int tid);
  void CreateThreadOperators(const framework::ProgramDesc& program);
  void SetRootScope(Scope* g_scope);
  void SetDevice();
  virtual void AddFidSet();
  void SetCommBatch(int comm_batch) { comm_batch_ = comm_batch; }
  void AddTrainFile(const std::string& filename);
  void SetMainProgram(const ProgramDesc& main_program_desc);
  void SetPlace(const paddle::platform::Place& place);
  void SetMaxTrainingEpoch(const int max_epoch);
  void BindingDataFeedMemory();
  void SetModelPrefix(const std::string& prefix) { model_prefix_ = prefix; }
  void SetInspectVarName(const std::string& inspect_var_name);
  void SetModelParamNames(const std::vector<std::string>& param_names);
  void SetSparseCommData(const std::map<std::string, int>& param_names);
  void SetDataFeed(const std::shared_ptr<DataFeed>& datafeed);
  void Train();
  virtual const char* PickOneFile();
  void UpdateEpochNum();

  virtual void SetDenseCommTensor(
      const std::vector<std::string>& param_names) {}
  virtual void Initialize() {}

 public:
  static std::mutex s_locker_for_pick_file_;
  static unsigned int s_current_file_idx_;
  static size_t s_current_finished_file_cnt_;
  static unsigned int s_current_epoch_;
  static int s_current_save_epoch_;
  static std::vector<std::string> s_thread_filelist_;   // filelist
  static bool s_is_first_worker_;

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
  std::unique_ptr<framework::ProgramDesc> main_program_;

  // binary data reader
  std::shared_ptr<DataFeed> local_reader_;

  std::string inspect_var_name_;
  std::vector<std::string> model_param_names_;
  std::map<std::string, int> sparse_comm_data_;

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
  static std::unique_ptr<ProgramDesc> LoadDescFromFile(
                                          const std::string& filename);
  void InitRootScope(Scope* scope);
  void SetInspectVarName(const std::string& inspect_var_name);
  void SetParamNames(const std::vector<std::string>& param_names);
  void SetMaxTrainingEpoch(const int max_epoch);
  Scope* GetRootScope() { return root_scope_; }
  void SetThreadNum(const int thread_num);
  void SetBatchSize(const int batch_size) { batch_size_ = batch_size; }
  void SetFileList(const char* filelist);
  void SetFileList(const std::vector<std::string> filelist);
  void SetDataFeedName(const char* feedname);

  void SetDataFeedParam(const datafeed::DataFeedParameter& feed_param) {
    data_feed_param_ = feed_param;
  }

  void SetCommBatch(int comm_batch) {
    comm_batch_ = comm_batch;
  }

  void SetModelPrefix(const std::string& model_prefix);
  void SetDenseCommTensor(const std::vector<std::string>& dense_comm_tensor);
  void SetSparseCommTensor(
      const std::vector<std::string>& sparse_comm_tensor);
  void SetSparseCommData(const std::map<std::string, int>& sparse_comm_data);
  virtual void PrepareThreads(const framework::ProgramDesc& host_program);
  void RunStartupProgram(const framework::ProgramDesc& program,
      framework::Scope* scope);
  void RunAsyncExecutor(const ProgramDesc& host_program);

 public:
  unsigned int thread_num_;
  datafeed::DataFeedParameter data_feed_param_;
  int max_epoch_;
  int batch_size_;
  int comm_batch_;
  std::vector<std::shared_ptr<ExecutorThreadWorker> > workers_;
  std::vector<std::thread> threads_;
  std::vector<std::string> filelist_;
  std::string inspect_var_name_;
  std::vector<std::string> model_param_names_;
  std::vector<std::string> dense_comm_tensor_;
  std::vector<std::string> sparse_comm_tensor_;
  std::map<std::string, int> sparse_comm_data_;
  std::string model_prefix_;
  std::string feed_name_;
  Scope* root_scope_;
  platform::Place place_;
};

}  // namespace framework
}  // namespace paddle
#endif  // PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
