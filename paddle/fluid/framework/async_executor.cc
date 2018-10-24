/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/async_executor.h"
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace framework {
std::mutex ExecutorThreadWorker::_s_locker_for_pick_file;
unsigned int ExecutorThreadWorker::_s_current_file_idx = 0;
size_t ExecutorThreadWorker::_s_current_finished_file_cnt = 0;
unsigned int ExecutorThreadWorker::_s_current_epoch = 0;
int ExecutorThreadWorker::_s_current_save_epoch = 0;
bool ExecutorThreadWorker::_s_is_first_worker = false;
std::vector<std::string> ExecutorThreadWorker::_s_thread_filelist;

void CreateTensor(Variable* var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<Scope>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, CHANNEL, RAW]",
        var_type);
  }
}

static void read_binary_file(const std::string& filename,
                             std::string* content) {
  std::string &contents = *content;
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  if (!fin.good()) {
    LOG(ERROR) << "Cannot open file " << filename.c_str();
  }
  fin.seekg(0, std::ios::end);
  contents.clear();
  contents.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&contents[0], contents.size());
  fin.close();
}

static void save_model(
    const std::unique_ptr<ProgramDesc> & main_program,
    Scope* scope,
    const std::vector<std::string> & param_names,
    const std::string & model_name,
    bool save_combine) {
  auto place = platform::CPUPlace();
  const BlockDesc& global_block = main_program->Block(0);
  std::vector<std::string> paralist;

  for (auto* var : global_block.AllVars()) {
    bool is_model_param = false;
    for (auto param_name : param_names) {
      if (var->Name() == param_name) {
        is_model_param = true;
        break;
      }
    }

    if (!is_model_param)  continue;

    if (!save_combine) {
      LOG(ERROR) << "model var name: " << var->Name().c_str();

      paddle::framework::AttributeMap attrs;
      attrs.insert({"file_path", model_name + "/" + var->Name()});
      auto save_op = paddle::framework::OpRegistry::CreateOp(
                                                      "save",
                                                      {{"X", {var->Name()}}},
                                                      {},
                                                      attrs);

      save_op->Run(*scope, place);
    } else {
      paralist.push_back(var->Name());
    }
  }

  if (save_combine) {
    std::sort(paralist.begin(), paralist.end());
    paddle::framework::AttributeMap attrs;
    attrs.insert({"file_path", model_name});
    auto save_op = paddle::framework::OpRegistry::CreateOp(
                                                      "save_combine",
                                                      {{"X", paralist}},
                                                      {},
                                                      attrs);
    save_op->Run(*scope, place);
  }
}   // end save_model


void ExecutorThreadWorker::add_train_file(const std::string& file) {
  _s_thread_filelist.push_back(file);
}

void ExecutorThreadWorker::create_thread_operators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  _op_names.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    _op_names.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    _ops.push_back(local_op_ptr);
    continue;
  }
}

void ExecutorThreadWorker::create_thread_scope(const ProgramDesc& program) {
  auto& block = program.Block(0);
  _thread_scope = &_root_scope->NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = _root_scope->Var(var->Name());
      CreateTensor(ptr, var->GetType());
      // LOGERR("create Persistable var[%s] finished",
      //      var->Name().c_str());
    } else {
      auto* ptr = _thread_scope->Var(var->Name());
      CreateTensor(ptr, var->GetType());
      // LOGERR("create unpersistable var[%s] finished",
      //      var->Name().c_str());
    }
  }
}

void ExecutorThreadWorker::set_datafeed(const std::shared_ptr<DataFeed>& datafeed) {
  _local_reader = datafeed;
}

void ExecutorThreadWorker::binding_datafeed_memory() {
  const std::vector<std::string>& input_feed = _local_reader->get_use_slot_alias();
  for (auto name : input_feed) {
    _local_reader->add_feed_var(_thread_scope->Var(name), name);
  }
}

void ExecutorThreadWorker::set_inspect_var_name(
    const std::string& inspect_var_name) {
  _inspect_var_name = inspect_var_name;
}

void ExecutorThreadWorker::set_model_param_names(
    const std::vector<std::string>& param_names) {
  _model_param_names = param_names;
}

void ExecutorThreadWorker::set_sparse_comm_data(
    const std::map<std::string, int>& param_names) {
  _sparse_comm_data = param_names;
}

void ExecutorThreadWorker::set_device() {
  static unsigned priority[] = {
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47
  };

  unsigned int i = this->_thread_id;

  if (i < sizeof(priority) / sizeof(unsigned)) {
    unsigned proc = priority[i];

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(proc, &mask);

    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
      LOG(ERROR) << "WARNING: Failed to set thread affinity for thread " << i;
    } else {
      CPU_ZERO(&mask);
      if ((0 == sched_getaffinity(0, sizeof(mask), &mask))
          && CPU_ISSET(proc, &mask)) {
        LOG(ERROR) << "TRACE: Thread " << i << " is running on processor " << proc << "...";
      }
    }
  }
}

void ExecutorThreadWorker::update_epoch_num() {
  _s_current_finished_file_cnt++;

  if (_s_current_finished_file_cnt >= _s_thread_filelist.size()) {
    _s_current_finished_file_cnt = 0;
    _s_current_epoch++;
  }
}

const char* ExecutorThreadWorker::pick_one_file() {
  std::string file_to_be_preocessed;
  std::lock_guard<std::mutex> lock(_s_locker_for_pick_file);

  if (_s_current_file_idx >= _s_thread_filelist.size()) {
    std::random_shuffle(_s_thread_filelist.begin(),
    _s_thread_filelist.end());
    _s_current_file_idx = 0;
    // _s_current_epoch++; //example: when one file, one thread, it's bug
    LOG(ERROR) << "thread " << _thread_id
               << ": finish traing for epoch " << _s_current_epoch + 1;
  }
  file_to_be_preocessed = _s_thread_filelist[_s_current_file_idx];

  _s_current_file_idx++;
  return file_to_be_preocessed.c_str();
}

void ExecutorThreadWorker::train() {
  LOG(ERROR) << "begin to train";
  set_device();
#ifdef LOCAL_PROF
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  // int total_batch = 0;
  for (auto& op : _ops) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(_ops.size());
  for (int i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
#endif
  std::string inspect_key = "inspect";
  if (!_inspect_var_name.empty()) {
    inspect_key = _inspect_var_name.substr(0,
                                          _inspect_var_name.find_first_of('_'));
  }

  for (unsigned i = 0; i < _max_epoch; ++i) {
    LOG(ERROR) << "epoch: " << i;
#ifdef LOCAL_PROF
    Timer timeline;
    double total_time = 0.0;
    double read_time = 0.0;
#endif
    float total_inspect = 0;
    int batch_num = 1;
    while (i == _s_current_epoch) {
      const char* filename = pick_one_file();
      _local_reader->set_file(filename);
      while (true) {
#ifdef LOCAL_PROF
        timeline.start();
#endif
        bool flag = _local_reader->read_batch();
        if (!flag) {
          break;
        }
#ifdef LOCAL_PROF
        timeline.pause();
        read_time += timeline.elapsed_sec();
        total_time += timeline.elapsed_sec();
#endif
        if (!flag) {
          break;
        }

        for (unsigned int i = 0; i < _ops.size(); ++i) {
#ifdef LOCAL_PROF
          timeline.start();
#endif
          _ops[i]->Run(*_thread_scope, _place);
#ifdef LOCAL_PROF
          timeline.pause();
          op_total_time[i] += timeline.elapsed_sec();
          total_time += timeline.elapsed_sec();
#endif
        }
        batch_num++;
        float avg_inspect = 0.0;
        if (!_inspect_var_name.empty()) {
          avg_inspect = _thread_scope->FindVar(_inspect_var_name)
                                     ->GetMutable<LoDTensor>()
                                     ->data<float>()[0];
        }
        total_inspect += avg_inspect;
        _thread_scope->DropKids();
      }
      update_epoch_num();
      LOG(ERROR) << "memory used after epoch " << i + 1
                 << " called: " << memory::memory_usage(_place);

#ifdef LOCAL_PROF
      for (int i = 0; i < op_total_time.size(); ++i) {
        std::cerr << "op_name:[" << i << "][" << op_name[i] << "]"
                  << " op_mean_time:[" << op_total_time[i] << "s]"
                  << std::endl;
      }
      std::cerr << "read time: " << read_time << "s" << std::endl;
#endif
    }
#ifdef LOCAL_PROF
    LOG(ERROR) << "mean " << inspect_key.c_str()
               << " of epoch " << i + 1 << ": " << total_inspect / batch_num
               << ", total_time: " << total_time;
#else
    LOG(ERROR) << "mean " << inspect_key.c_str()
               << " of epoch " << i + 1 << ": " << total_inspect / batch_num;
#endif
    if (_thread_id == 0) {
      char modelfile[1024];
      snprintf(&modelfile[0],
              sizeof(modelfile),
              "%s_epoch%d.model",
              _model_prefix.c_str(),
              i);
      std::string model_filename = std::string(modelfile);
      // this save_inference_model can only save imdbtask, should make this
      // general
      //
      // currently comment it
      LOG(ERROR) << "Going to save model " << modelfile;
      save_model(_main_program,
          _thread_scope,
          _model_param_names,
          model_filename,
          true);
    }
  }
}

void ExecutorThreadWorker::set_thread_id(int tid) {
  _thread_id = tid;
}

void ExecutorThreadWorker::set_place(const platform::Place& place) {
  _place = place;
}

void ExecutorThreadWorker::set_main_program(
    const ProgramDesc& main_program_desc) {
  _main_program.reset(new ProgramDesc(main_program_desc));
}

void ExecutorThreadWorker::set_root_scope(Scope* g_scope) {
  _root_scope = g_scope;
}

void ExecutorThreadWorker::set_max_training_epoch(int max_epoch) {
  _max_epoch = max_epoch;
}

MultiExecutor::MultiExecutor(const platform::Place& place) : _place(place) {}

void MultiExecutor::init_root_scope(Scope* scope) {
  _root_scope = scope;
}

void MultiExecutor::set_max_training_epoch(int max_epoch) {
  _max_epoch = max_epoch;
}

void MultiExecutor::set_datafeed_name(const char* feedname) {
  _feed_name = std::string(feedname);
}

void MultiExecutor::set_model_prefix(const std::string& model_prefix) {
  _model_prefix = model_prefix;
}

void MultiExecutor::run_startup_program(const ProgramDesc& program,
                                        Scope* scope) {
  auto& block = program.Block(0);
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = scope->Var(var->Name());
      CreateTensor(ptr, var->GetType());
      // LOGERR("Persistable Var Name:%s", var->Name().c_str());
    }
  }

  std::map<std::string, int> param_dict;
  std::vector<OperatorBase *> ops;
  for (auto& op_desc : block.AllOps()) {
    std::vector<std::string> param_name_vec = op_desc->OutputArgumentNames();
    bool need_to_run = false;
    for (auto& name : param_name_vec) {
      if (param_dict.find(name) == param_dict.end()) {
        param_dict[name] = 1;
        need_to_run = true;
      }
    }
    if (need_to_run) {
      std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
      OperatorBase* local_op_ptr = local_op.release();
      ops.push_back(local_op_ptr);
    }
  }
  // LOGERR("There are %d parameters in startup program, %d op needs to run",
  //        param_dict.size(), ops.size());

  for (auto& op : ops) {
    op->Run(*scope, _place);
  }
  // LOGERR("total time for startup program: %fs", timeline.elapsed_sec());
  for (auto& op : ops) {
    delete op;
  }
  // LOGERR("run startup program done.");
}

std::unique_ptr<ProgramDesc> MultiExecutor::load_desc_from_file(
    const std::string& f) {
  std::string program_desc_str;
  read_binary_file(f, &program_desc_str);
  std::unique_ptr<ProgramDesc> program(new ProgramDesc(program_desc_str));
  return program;
}

void MultiExecutor::set_dense_comm_tensor(
    const std::vector<std::string>& dense_comm_tensor) {
  _dense_comm_tensor.resize(dense_comm_tensor.size());
  for (unsigned int i = 0; i < dense_comm_tensor.size(); ++i) {
    _dense_comm_tensor[i] = dense_comm_tensor[i];
  }
}

void MultiExecutor::set_sparse_comm_tensor(
    const std::vector<std::string>& sparse_comm_tensor) {
  _sparse_comm_tensor.resize(sparse_comm_tensor.size());
  for (unsigned int i = 0; i < sparse_comm_tensor.size(); ++i) {
    _sparse_comm_tensor[i] = sparse_comm_tensor[i];
  }
}

void MultiExecutor::set_sparse_comm_data(
    const std::map<std::string, int>& sparse_comm_data) {
  _sparse_comm_data = sparse_comm_data;
  LOG(INFO) << "Sparse comm data: " << _sparse_comm_data.size();
}

void MultiExecutor::set_filelist(const char* filelist) {
  _filelist.clear();
  std::ifstream fin(filelist);
  std::string filename;
  while (fin >> filename) {
    LOG(ERROR) << "add " << filename.c_str() << " to filelist";
    _filelist.push_back(filename);
  }
  fin.close();
}

void MultiExecutor::set_filelist(std::vector<std::string> tfiles) {
  _filelist.clear();
  _filelist.insert(_filelist.end(), tfiles.begin(), tfiles.end());
  return;
}

void MultiExecutor::set_inspect_var_name(const std::string& inspect_var_name) {
  _inspect_var_name = inspect_var_name;
}

void MultiExecutor::set_param_names(const std::vector<std::string>& param_names) {
  _model_param_names = param_names;
}

void MultiExecutor::set_thread_num(const int thread_num) {
  _thread_num = thread_num;
}

void MultiExecutor::prepare_threads(const ProgramDesc& host_program) {
  _workers.resize(_thread_num);
  for (unsigned i = 0; i < _thread_num; ++i) {
    _workers[i].reset(new ExecutorThreadWorker);
    _workers[i]->set_thread_id(i);
    _workers[i]->create_thread_operators(host_program);
    _workers[i]->set_root_scope(_root_scope);
    _workers[i]->set_place(_place);
    _workers[i]->set_max_training_epoch(_max_epoch);
    _workers[i]->create_thread_scope(host_program);
    _workers[i]->set_inspect_var_name(_inspect_var_name);
    _workers[i]->set_model_param_names(_model_param_names);
    _workers[i]->set_sparse_comm_data(_sparse_comm_data);
    _workers[i]->set_main_program(host_program);
    _workers[i]->set_model_prefix(_model_prefix);
  }

  for (unsigned i = 0; i < _filelist.size(); ++i) {
    // suppose at least one trainer thread here, and
    // filelist is static so that we only add filelist once
    _workers[0]->add_train_file(_filelist[i]);
  }
  // mpi_wrapper::ModelParam model_param(true);
  // _workers[0]->register_parallel_training_param(model_param);

  for (unsigned i = 0; i < _thread_num; ++i) {
    // new a datafeed here
    std::shared_ptr<DataFeed> local_feed = create_datafeed(_feed_name.c_str());
    local_feed->init(_data_feed_param);
    local_feed->set_batch_size(_batch_size);
    _workers[i]->set_datafeed(local_feed);
    _workers[i]->binding_datafeed_memory();
    _workers[i]->set_thread_id(i);
  }
}

void MultiExecutor::run_multi_executor(const ProgramDesc& host_program) {
  // thread binding here?
  prepare_threads(host_program);
  for (unsigned i = 0; i < _thread_num; ++i) {
    _threads.push_back(std::thread(&ExecutorThreadWorker::train,
                      _workers[i].get()));
  }

  for (auto& th : _threads) {
    th.join();
  }
}

}   // end namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
