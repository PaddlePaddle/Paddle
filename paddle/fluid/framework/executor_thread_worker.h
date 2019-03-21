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
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#endif

namespace paddle {
namespace framework {

void CreateTensor(Variable* var, proto::VarType::Type var_type);
#ifdef PADDLE_WITH_PSLIB
static const uint32_t MAX_FEASIGN_NUM = 1000 * 100 * 100;

struct AsyncWorkerParamConfig {
  int slot_dim;
  int fea_dim;
  int32_t tmp_push_dense_wait_times;
  int32_t tmp_push_sparse_wait_times;

  std::vector<std::string> skip_op;

  std::map<uint64_t, std::vector<std::string>> dense_variable_name;
  std::map<uint64_t, std::vector<std::string>> dense_gradient_variable_name;
  std::vector<int> dense_table_id;
  // fea_dim for each dense table
  std::vector<uint32_t> dense_table_size;
  std::vector<int> sparse_table_id;
  std::map<uint64_t, std::vector<std::string>> slot_input_vec;
  std::map<uint64_t, std::vector<std::string>> gradient_var;
  std::map<std::string, uint64_t> slot_alias_to_table;
};

struct DensePullThreadParam {
  std::shared_ptr<paddle::ps::PSClient> ps_client;
  int threshold;
  int training_thread_num;
  Scope* root_scope;
  std::map<uint64_t, std::vector<std::string>>* dense_params;
  int sleep_time_ms = 2;
};

class DensePullThread {
 public:
  explicit DensePullThread(const DensePullThreadParam& param)
      : _running(false) {
    _ps_client = param.ps_client;
    _threshold = param.threshold;
    _thread_num = param.training_thread_num;
    _root_scope = param.root_scope;
    _sleep_time_ms = param.sleep_time_ms;

    for (auto& t : *param.dense_params) {
      _dense_variable_name[t.first].insert(_dense_variable_name[t.first].end(),
                                           t.second.begin(), t.second.end());
      _training_versions[t.first].resize(_thread_num, 0);
      _last_versions[t.first] = 0;
      _current_version[t.first] = 0;
    }
  }

  int start();

  void stop() {
    if (_running) {
      _running = false;
      _t.join();
    }
  }

  void increase_thread_version(int thread_id, uint64_t table_id);
  void reset_thread_version(uint64_t table_id);
  std::future<int32_t> pull_dense(uint64_t table_id);
  void pull_dense2(uint64_t table_id);
  void wait_all();

 private:
  void run();
  bool check_update_param(uint64_t table_id);

 private:
  std::shared_ptr<paddle::ps::PSClient> _ps_client;
  int _thread_num;
  int _threshold;
  int _sleep_time_ms;
  Scope* _root_scope;
  bool _running;

  std::map<uint64_t, uint64_t> _last_versions;
  std::map<uint64_t, uint64_t> _current_version;
  std::mutex _mutex_for_version;
  std::map<uint64_t, std::vector<uint64_t>> _training_versions;
  std::map<uint64_t, std::vector<std::string>> _dense_variable_name;

  std::thread _t;

  std::vector<::std::future<int32_t>> _pull_dense_status;

  std::map<uint64_t, std::vector<paddle::ps::Region>> _regions;
  uint32_t _pull_dense_fail_times = 0;

  std::vector<float> _base_norm_param;
  std::vector<float> _mean;
  std::vector<float> _scale;
  float _squared_sum_epsilon = 1e-4;
  std::mutex _mutex_for_mean_scale;

  float _total_batch_num = 0;
};
#endif

class ExecutorThreadWorker {
 public:
  ExecutorThreadWorker()
      : thread_id_(-1), root_scope_(NULL), thread_scope_(NULL), debug_(false) {}
  virtual ~ExecutorThreadWorker() {}

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
  virtual void TrainFiles();
  // with timer log
  virtual void TrainFilesWithTimer();
  // set fetch variable names from python interface assigned by users
  void SetFetchVarNames(const std::vector<std::string>& fetch_var_names);
#ifdef PADDLE_WITH_PSLIB
  virtual void SetPSlibPtr(
      std::shared_ptr<paddle::distributed::PSlib> pslib_ptr) {}
  virtual void SetPullDenseThread(std::shared_ptr<DensePullThread> dpt) {}
  virtual void SetParamConfig(AsyncWorkerParamConfig* param_config) {}
#endif

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
  std::vector<std::string> fetch_var_names_;
  std::vector<std::vector<float>> fetch_values_;
  bool debug_;
};

#ifdef PADDLE_WITH_PSLIB
class AsyncExecutorThreadWorker : public ExecutorThreadWorker {
 public:
  AsyncExecutorThreadWorker() {}
  virtual ~AsyncExecutorThreadWorker() {}
  void SetPSlibPtr(std::shared_ptr<paddle::distributed::PSlib> pslib_ptr);
  void SetPullDenseThread(std::shared_ptr<DensePullThread> dpt);
  void SetParamConfig(AsyncWorkerParamConfig* param_config);
  void TrainFiles();
  void TrainOneNetwork();
  void PrepareParams();
  void UpdateParams();
  void PullSparse(int table_id);
  void FillSparse(int table_id);
  void PushSparse(int table_id);
  void PushDense(int table_id);

  void check_pull_push_memory(const std::vector<uint64_t>& features,
                              std::vector<float*>* push_g, int dim);
  void check_pull_push_memory(const std::vector<uint64_t>& features,
                              std::vector<std::vector<float>>* push_g, int dim);
  void collect_feasign_info(int table_id);

 private:
  struct FeasignInfo {
    uint32_t slot;
    uint32_t ins;
    int64_t label;
  };

  std::map<uint64_t, std::vector<uint64_t>> _features;
  std::map<uint64_t, std::vector<FeasignInfo>> _fea_info;
  std::map<uint64_t, std::vector<std::vector<float>>> _feature_value;
  std::map<uint64_t, std::vector<std::vector<float>>> _feature_push_value;

  std::shared_ptr<paddle::distributed::PSlib> _pslib_ptr;

  std::shared_ptr<DensePullThread> _pull_dense_thread;

  std::vector<::std::future<int32_t>> _pull_sparse_status;
  std::vector<::std::future<int32_t>> _pull_dense_status;
  std::vector<::std::future<int32_t>> _push_sparse_status;
  std::vector<::std::future<int32_t>> _push_dense_status;

  AsyncWorkerParamConfig* _param_config;
};
#endif

}  // namespace framework
}  // namespace paddle
