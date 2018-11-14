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
#include <sys/types.h>
#include <sys/stat.h>
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
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace framework {

bool AsyncExecutor::workers_initialized_ = false;

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

static void ReadBinaryFile(const std::string& filename,
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

static void SaveModel(
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
}   // end SaveModel

void ExecutorThreadWorker::Reset() {
  inspect_values_.clear();
}
void ExecutorThreadWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
}

void ExecutorThreadWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);
  thread_scope_ = &root_scope_->NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = root_scope_->Var(var->Name());
      CreateTensor(ptr, var->GetType());
    } else {
      auto* ptr = thread_scope_->Var(var->Name());
      CreateTensor(ptr, var->GetType());
    }
  }
}

void ExecutorThreadWorker::SetDataFeed(DataFeed& datafeed) {
  if (typeid(datafeed) == typeid(TextClassDataFeed)) {
    local_reader_.reset(
        new TextClassDataFeed(dynamic_cast<TextClassDataFeed &>(datafeed)));
    local_reader_->SetThreadId(thread_id_);
  }
}

void ExecutorThreadWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed = local_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    local_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void ExecutorThreadWorker::SetInspectVarNames(
    const std::vector<std::string>& inspect_var_names) {
  inspect_var_names_.clear();
  inspect_var_names_.insert(inspect_var_names_.end(),
                            inspect_var_names.begin(), inspect_var_names.end());
}

void ExecutorThreadWorker::SetModelParamNames(
    const std::vector<std::string>& param_names) {
  model_param_names_ = param_names;
}

void ExecutorThreadWorker::SetDevice() {
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

  unsigned int i = this->thread_id_;

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
        LOG(ERROR) << "TRACE: Thread " << i
                   << " is running on processor " << proc
                   << "...";
      }
    }
  }
}


void ExecutorThreadWorker::Train() {
  LOG(ERROR) << "begin to train";
  SetDevice();

  int inspect_var_num = inspect_var_names_.size();
  inspect_values_.clear();
  inspect_values_.resize(inspect_var_num, 0);

  local_reader_->WaitNextEpoch();
  int epoch = local_reader_->GetCurrentEpoch();

  LOG(ERROR) << "epoch: " << epoch;

  int batch_num = 1;

  while (true) {
    const char *file = local_reader_->PickOneFile();
    if (file == NULL) {
      break;
    }

    if (!local_reader_->SetFile(file)) {
      break;
    }

    while (true) {
      bool flag = local_reader_->ReadBatch();
      if (!flag) {
        break;
      }

      for (unsigned int i = 0; i < ops_.size(); ++i) {
        ops_[i]->Run(*thread_scope_, place_);
      }
      batch_num++;

      float avg_inspect = 0.0;
      for (int i = 0; i < inspect_var_num; ++i) {
        avg_inspect = thread_scope_->FindVar(inspect_var_names_[i])
                                   ->GetMutable<LoDTensor>()
                                   ->data<float>()[0];
        inspect_values_[i] += avg_inspect;
      }
      thread_scope_->DropKids();
    }

    local_reader_->UpdateEpochNum();
    LOG(ERROR) << "memory used after epoch " << epoch + 1
               << " called: " << memory::memory_usage(place_);
  }

  for (int i = 0; i < inspect_var_num; ++i) {
    inspect_values_[i] /= batch_num;
    std::string var = inspect_var_names_[i].substr(
                          0,
                          inspect_var_names_[i].find_first_of("_"));
    LOG(ERROR) << "mean " << var.c_str()
               << " of epoch " << i + 1 << ": " << inspect_values_[i];
  }

  if (thread_id_ == 0) {
    char modelfile[1024];
    snprintf(&modelfile[0], sizeof(modelfile), "%s_epoch%d.model",
             model_prefix_.c_str(), epoch);
    std::string model_filename = std::string(modelfile);
    // this save_inference_model can only save imdbtask, should make this
    // general
    //
    // currently comment it
    LOG(ERROR) << "Going to save model " << modelfile;
    SaveModel(main_program_,
              thread_scope_,
              model_param_names_,
              model_filename,
              true);
  }
}

void ExecutorThreadWorker::SetThreadId(int tid) {
  thread_id_ = tid;
}

void ExecutorThreadWorker::SetPlace(const platform::Place& place) {
  place_ = place;
}

void ExecutorThreadWorker::SetMainProgram(
    const ProgramDesc& main_program_desc) {
  main_program_.reset(new ProgramDesc(main_program_desc));
}

void ExecutorThreadWorker::SetRootScope(Scope* g_scope) {
  root_scope_ = g_scope;
}

void ExecutorThreadWorker::SetMaxTrainingEpoch(int max_epoch) {
  max_epoch_ = max_epoch;
}

AsyncExecutor::AsyncExecutor(ProgramDesc& main_program,
                       const std::vector<std::string>& param_names,
                       TextClassDataFeed& data_feed,
                       unsigned int thread_num,
                       const platform::Place& place)
    : thread_num_(thread_num),
      place_(place),
      main_program_(main_program),
      data_feed_(data_feed) {
  model_param_names_.clear();
  model_param_names_.insert(model_param_names_.end(),
                            param_names.begin(),
                            param_names.end());
}

void AsyncExecutor::InitRootScope(Scope* scope) {
  root_scope_ = scope;
}

void AsyncExecutor::SetMaxTrainingEpoch(int max_epoch) {
  max_epoch_ = max_epoch;
}

void AsyncExecutor::SetModelPrefix(const std::string& model_prefix) {
  model_prefix_ = model_prefix;
}

void AsyncExecutor::RunStartupProgram(const ProgramDesc& program,
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
    op->Run(*scope, place_);
  }
  // LOGERR("total time for startup program: %fs", timeline.elapsed_sec());
  for (auto& op : ops) {
    delete op;
  }
  // LOGERR("run startup program done.");
}

std::unique_ptr<ProgramDesc> AsyncExecutor::LoadDescFromFile(
    const std::string& f) {
  std::string program_desc_str;
  ReadBinaryFile(f, &program_desc_str);
  std::unique_ptr<ProgramDesc> program(new ProgramDesc(program_desc_str));
  return program;
}

void AsyncExecutor::SetInspectVarNames(
    const std::vector<std::string>& inspect_var_names) {
  inspect_var_names_.clear();
  inspect_var_names_.insert(inspect_var_names_.end(),
                            inspect_var_names.begin(), inspect_var_names.end());
}

void AsyncExecutor::PrepareThreads(const ProgramDesc& host_program) {
  workers_.resize(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i].reset(new ExecutorThreadWorker);
    workers_[i]->SetThreadId(i);
    workers_[i]->CreateThreadOperators(host_program);
    workers_[i]->SetRootScope(root_scope_);
    workers_[i]->SetPlace(place_);
    workers_[i]->SetMaxTrainingEpoch(max_epoch_);
    workers_[i]->CreateThreadScope(host_program);
    workers_[i]->SetInspectVarNames(inspect_var_names_);
    workers_[i]->SetModelParamNames(model_param_names_);
    workers_[i]->SetMainProgram(host_program);
    workers_[i]->SetModelPrefix(model_prefix_);
    //
    // new a datafeed here
    workers_[i]->SetDataFeed(data_feed_);
    workers_[i]->BindingDataFeedMemory();
  }
}

std::vector<float>& AsyncExecutor::Run(
    const std::vector<std::string>& inspect_var_names) {
  SetInspectVarNames(inspect_var_names);
  threads_.clear();

  // thread binding here?
  if (workers_initialized_ == false) {
    PrepareThreads(main_program_);
    workers_initialized_ = true;
  }

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->Reset();
    workers_[i]->SetInspectVarNames(inspect_var_names);
    threads_.push_back(std::thread(&ExecutorThreadWorker::Train,
                      workers_[i].get()));
  }

  for (auto& th : threads_) {
    th.join();
  }

  inspect_values_.clear();
  inspect_values_.resize(inspect_var_names_.size(), 0);


  std::vector<std::vector<float>*> inspect_value_vectors;
  inspect_value_vectors.resize(thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    inspect_value_vectors[i] = &workers_[i]->GetInspectValues();
  }

  for (unsigned int i = 0; i < inspect_var_names_.size(); ++i) {
    float value = 0.0;
    for (int j = 0; j < thread_num_; ++j) {
      value += inspect_value_vectors[j]->at(i);
    }
    value /= thread_num_;
    inspect_values_[i] = value;
  }

  return inspect_values_;
}

void AsyncExecutor::LoadInitModel() {
  auto place = paddle::platform::CPUPlace();
  auto* executor = new paddle::framework::Executor(place);

  std::string init_prog_file = model_path_ + "/" + init_prog_file_;
  std::string init_model_file = model_path_ + "/" + init_model_file_;

  struct stat stat_buf;

  if (stat(init_prog_file.c_str(), &stat_buf) == 0 &&
      S_ISREG(stat_buf.st_mode) &&
      stat(init_model_file.c_str(), &stat_buf) == 0 &&
      S_ISREG(stat_buf.st_mode)) {
    paddle::inference::Load(executor,
                          GetRootScope(),
                          model_path_ + "/" + init_prog_file_,
                          model_path_ + "/" + init_model_file_);
  }
}
}   // einit_modelnd namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
