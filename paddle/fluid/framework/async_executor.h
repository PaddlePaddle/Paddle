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
  void create_thread_scope(const framework::ProgramDesc& program);
  void set_datafeed(const DataFeed& datafeed);
  void set_thread_id(int tid);
  void create_thread_operators(const framework::ProgramDesc& program);
  void set_root_scope(Scope* g_scope);
  void set_device();
  virtual void add_fid_set();
  void set_comm_batch(int comm_batch) { _comm_batch = comm_batch; }
  void add_train_file(const std::string& filename);
  void set_main_program(const ProgramDesc& main_program_desc);
  void set_place(const paddle::platform::Place& place);
  void set_max_training_epoch(const int max_epoch);
  void binding_datafeed_memory();
  void set_model_prefix(const std::string& prefix) { _model_prefix = prefix; }
  void set_inspect_var_name(const std::string& inspect_var_name);
  void set_model_param_names(const std::vector<std::string>& param_names);
  void set_sparse_comm_data(const std::map<std::string, int>& param_names);
  void set_datafeed(const std::shared_ptr<DataFeed>& datafeed);
  virtual void mpi_train();
  void gpu_train();
  void train();
  virtual const char* pick_one_file();
  void update_epoch_num();

  virtual void set_dense_comm_tensor(
      const std::vector<std::string>& param_names) {}
  virtual void initialize() {}

 public:
  static std::mutex _s_locker_for_pick_file;
  static unsigned int _s_current_file_idx;
  static size_t _s_current_finished_file_cnt;
  static unsigned int _s_current_epoch;
  static int _s_current_save_epoch;
  static std::vector<std::string> _s_thread_filelist;   // filelist
  static bool _s_is_first_worker;

 protected:
  // thread index
  int _thread_id;

  // current training file
  int _cur_fileidx;

  // max epoch for each thread
  unsigned int _max_epoch;

  // instances learned currently
  int _comm_batch;
  std::string _model_prefix;
  std::vector<std::string> _op_names;

  // local ops for forward and backward
  std::vector<OperatorBase *> _ops;

  // main program for training
  std::unique_ptr<framework::ProgramDesc> _main_program;

  // binary data reader
  std::shared_ptr<DataFeed> _local_reader;

  std::string _inspect_var_name;
  std::vector<std::string> _model_param_names;
  std::map<std::string, int> _sparse_comm_data;
  std::vector<int> _ids_buffer;

  // execution place
  platform::Place _place;

  // root scope for model parameters
  Scope* _root_scope;

  // a thread scope, father scope is global score which is shared
  Scope* _thread_scope;
};

class MultiExecutor {
 public:
  explicit MultiExecutor(const platform::Place& place);
  virtual ~MultiExecutor() {}
  static std::unique_ptr<ProgramDesc> load_desc_from_file(
                                          const std::string& filename);
  void init_root_scope(Scope* scope);
  void set_inspect_var_name(const std::string& inspect_var_name);
  void set_param_names(const std::vector<std::string>& param_names);
  void set_max_training_epoch(const int max_epoch);
  Scope* get_root_scope() { return _root_scope; }
  void set_thread_num(const int thread_num);
  void set_batch_size(const int batch_size) { _batch_size = batch_size; }
  void set_filelist(const char* filelist);
  void set_filelist(const std::vector<std::string> filelist);
  void set_datafeed_name(const char* feedname);

  void set_data_feed_param(const datafeed::DataFeedParameter& feed_param) {
    _data_feed_param = feed_param;
  }

  void set_comm_batch(int comm_batch) {
    _comm_batch = comm_batch;
  }

  void set_model_prefix(const std::string& model_prefix);
  void set_dense_comm_tensor(const std::vector<std::string>& dense_comm_tensor);
  void set_sparse_comm_tensor(
      const std::vector<std::string>& sparse_comm_tensor);
  void set_sparse_comm_data(const std::map<std::string, int>& sparse_comm_data);
  virtual void prepare_threads(const framework::ProgramDesc& host_program);
  void run_startup_program(const framework::ProgramDesc& program,
      framework::Scope* scope);
  void run_multi_executor(const ProgramDesc& host_program);

 public:
  unsigned int _thread_num;
  datafeed::DataFeedParameter _data_feed_param;
  int _max_epoch;
  int _batch_size;
  int _comm_batch;
  std::vector<std::shared_ptr<ExecutorThreadWorker> > _workers;
  std::vector<std::thread> _threads;
  std::vector<std::string> _filelist;
  std::string _inspect_var_name;
  std::vector<std::string> _model_param_names;
  std::vector<std::string> _dense_comm_tensor;
  std::vector<std::string> _sparse_comm_tensor;
  std::map<std::string, int> _sparse_comm_data;
  int node_num;
  std::string _model_prefix;
  ProgramDesc _host_program;
  std::string _feed_name;
  Scope* _root_scope;
  platform::Place _place;
};

}  // namespace framework
}  // namespace paddle
#endif  // PADDLE_FLUID_FRAMEWORK_ASYNC_EXECUTOR_H_
/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
