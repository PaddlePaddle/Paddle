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

void ExecutorThreadWorker::SetDataFeed(
    const std::shared_ptr<DataFeed>& datafeed) {
  local_reader_ = datafeed;
}

void ExecutorThreadWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed =
      thread_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    local_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void ExecutorThreadWorker::SetDevice() {
  // at most 48 threads binding currently
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
        LOG(ERROR) << "TRACE: Thread " << i <<
            " is running on processor " << proc << "...";
      }
    }
  }
}

void ExecutorThreadWorker::TrainFiles() {
  // todo: configurable
  SetDevice();
  thread_reader_->Start();
  while (int cur_batch = thread_reader_->Next()) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }
    thread_scope_->DropKids();
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

AsyncExecutor::AsyncExecutor(const platform::Place& place) : place_(place) {}

void AsyncExecutor::CreateThreads(const ExecutorThreadWorker* worker,
                                  const ProgramDesc& main_program,
                                  const DataFeed& reader,
                                  const Scope& root_scope,
                                  const int thread_index) {
  worker->SetThreadid(thread_index);
  worker->CreateThreadOperators(main_program);
  worker->CreateThreadScope(main_program);
  worker->SetDataFeed(reader);
  worker->BindingDataFeedMemory(reader);
  worker->SetMainProgram(main_program);
  worker->SetRootScope(root_scope);
}

shared_ptr<DataFeed> AsyncExecutor::CreateDataFeed(const char * feed_name) {
  if (g_datafeed_map.count(string(feed_name)) < 1) {
    return NULL;
  }
  return g_datafeed_map[feed_name]();
}

void AsyncExecutor::CheckFiles(
    const std::vector<std::string>& files) {
  // function for user to check file formats
  // should be exposed to users
}

/*
  in case there are binary files we want to train
  and in general this is the fastest way to train
  different calls allow thread_num to be different
  threads are created on the fly
  workers are created on the fly
  readers are created on the fly
  files are fed into readers on the fly
 */
/*
  class_name
  batch_size: max batch size
  use_slot: 
  queue_size: 
  buffer_size:
  padding_index: 
 */
void AsyncExecutor::RunFromFiles(
    const ProgramDesc& main_program,
    const DataFeedDesc& data_feed_desc,
    const std::vector<std::string> & files,
    const int thread_num) {
  // todo: remove fluid related interface
  root_scope_->DropKids();
  std::vector<std::thread> threads;
  threads.resize(thread_num);

  /*
    readerDesc: protobuf description for reader initlization
    argument: class_name, batch_size, use_slot, queue_size, buffer_size, padding_index
    
    reader: 
    1) each thread has a reader, reader will read input data and 
    put it into input queue
    2) each reader has a Next() iterface, that can fetch an instance
    from the input queue
   */
  // todo: should be factory method for creating datafeed
  std::vector<std::shared_ptr<DataFeed> > readers;
  readers.resize(thread_num);
  for (auto& reader : readers) {
    // create by factory name
    reader.reset(new DataFeed);
    reader.SetFileList(files);
  }

  std::vector<std::shared_ptr<ExecutorThreadWorker> > workers;
  workers.resize(thread_num);
  for (auto& worker : workers) {
    worker.reset(new ExecutorThreadWorker);
  }

  // prepare thread resource here
  for (int thidx = 0; thidx < thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program,
                  readers[thidx].get(), root_scope_, thidx);
  }
  
  // start executing ops in multiple threads
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    threads.push_back(std::thread(&ExecutorThreadWorker::TrainFiles,
                                  workers[thidx].get()));
  }

  for (auto& th : threads) {
    th.join();
  }
  // fetch variables in scope 0, and return
}

}   // einit_modelnd namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
