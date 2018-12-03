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
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"
namespace paddle {
namespace framework {

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
      root_scope_, "root_scope should be set before creating thread scope");

  thread_scope_ = &root_scope_->NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    } else {
      auto* ptr = thread_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
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
  fetch_var_names_.insert(fetch_var_names_.end(), fetch_var_names.begin(),
                          fetch_var_names.end());
}

void ExecutorThreadWorker::SetDevice() {
#if defined _WIN32 || defined __APPLE__
  return;
#else
  static unsigned concurrency_cap = std::thread::hardware_concurrency();
  int thread_id = this->thread_id_;

  if (thread_id < concurrency_cap) {
    unsigned proc = thread_id;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(proc, &mask);

    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
      VLOG(1) << "WARNING: Failed to set thread affinity for thread "
              << thread_id;
    } else {
      CPU_ZERO(&mask);
      if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) ||
          (CPU_ISSET(proc, &mask) == 0)) {
        VLOG(3) << "WARNING: Failed to set thread affinity for thread "
                << thread_id;
      }
    }
  } else {
    VLOG(1) << "WARNING: Failed to set thread affinity for thread "
            << thread_id;
  }
#endif
}

template <typename T>
void print_lod_tensor(std::string var_name, const LoDTensor& lod_tensor) {
  auto inspect = lod_tensor.data<T>();
  auto element_num = lod_tensor.numel();

  std::ostringstream sstream;
  sstream << var_name << " (element num " << element_num << "): [";
  sstream << inspect[0];
  for (int j = 1; j < element_num; ++j) {
    sstream << " " << inspect[j];
  }
  sstream << "]";

  std::cout << sstream.str() << std::endl;
}

void print_fetch_var(Scope* scope, std::string var_name) {
  const LoDTensor& tensor = scope->FindVar(var_name)->Get<LoDTensor>();

  if (std::type_index(tensor.type()) ==
      std::type_index(typeid(platform::float16))) {
    print_lod_tensor<platform::float16>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(float))) {
    print_lod_tensor<float>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(double))) {
    print_lod_tensor<double>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(int))) {
    print_lod_tensor<int>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int64_t))) {
    print_lod_tensor<int64_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(bool))) {
    print_lod_tensor<bool>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(uint8_t))) {
    print_lod_tensor<uint8_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int16_t))) {
    print_lod_tensor<int16_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int8_t))) {
    print_lod_tensor<int8_t>(var_name, tensor);
  } else {
    VLOG(1) << "print_fetch_var: unrecognized data type:"
            << tensor.type().name();
  }

  return;
}

void ExecutorThreadWorker::TrainFiles() {
  // todo: configurable
  SetDevice();

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num);

  thread_reader_->Start();

  int cur_batch;
  int batch_cnt = 0;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }

    ++batch_cnt;
    thread_scope_->DropKids();

    if (debug_ == false || thread_id_ != 0) {
      continue;
    }

    for (int i = 0; i < fetch_var_num; ++i) {
      print_fetch_var(thread_scope_, fetch_var_names_[i]);
    }  // end for (int i = 0...)
  }    // end while ()
}

void ExecutorThreadWorker::SetThreadId(int tid) { thread_id_ = tid; }

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

}  // einit_modelnd namespace framework
}  // end namespace paddle
