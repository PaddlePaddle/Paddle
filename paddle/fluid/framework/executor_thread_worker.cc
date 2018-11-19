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

#include "paddle/fluid/framework/executor_thread_worker.h"
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

void ExecutorThreadWorker::CreateThreadResource(
    const framework::ProgramDesc& program,
    const paddle::platform::Place& place) {
  CreateThreadScope(program);
  CreateThreadOperators(program);
  SetMainProgram(program);
  SetPlace(place);
}

void ExecutorThreadWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      "root_scope should be set before creating thread scope");

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
  thread_reader_ = datafeed;
}

void ExecutorThreadWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed =
      thread_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    thread_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void ExecutorThreadWorker::SetFetchVarNames(
    const std::vector<std::string>& fetch_var_names) {
  fetch_var_names_.clear();
  fetch_var_names_.insert(fetch_var_names_.end(),
                          fetch_var_names.begin(), fetch_var_names.end());
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

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num, 0);

  thread_reader_->Start();

  int cur_batch;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }

    float avg_inspect = 0.0;
    for (int i = 0; i < fetch_var_num; ++i) {
      avg_inspect = thread_scope_->FindVar(fetch_var_names_[i])
                                 ->GetMutable<LoDTensor>()
                                 ->data<float>()[0];
      fetch_values_[i] += avg_inspect;
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

}   // einit_modelnd namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
