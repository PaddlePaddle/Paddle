/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <signal.h>

#include "paddle/fluid/framework/executor_cache.h"
#include "paddle/fluid/operators/data/map_runner.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace operators {
namespace data {

MapRunner::MapRunner(
    const std::shared_ptr<BlockDesc> map_block, const int64_t program_id,
    const Scope* scope, const platform::Place& place,
    const std::vector<std::string>& input_var_names,
    const std::vector<std::string>& output_var_names,
    const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> input_queues,
    const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues)
    : running_(true),
      shutdown_(false),
      map_block_(map_block),
      program_id_(program_id),
      place_(place),
      input_var_names_(input_var_names),
      output_var_names_(output_var_names),
      input_queues_(input_queues),
      output_queues_(output_queues) {
  PADDLE_ENFORCE_EQ(
      input_var_names_.size(), input_queues_.size(),
      platform::errors::InvalidArgument(
          "input_var_names length should be equal to input_queues length, "
          "but recieve %d != %d.",
          input_var_names_.size(), input_queues_.size()));
  PADDLE_ENFORCE_EQ(
      output_var_names_.size(), output_queues_.size(),
      platform::errors::InvalidArgument(
          "output_var_names length should be equal to output_queues length, "
          "but recieve %d != %d.",
          output_var_names_.size(), output_queues_.size()));

  StartMapThread(scope);
}

bool MapRunner::ShareInputsIntoScope(Scope* scope) {
  for (size_t i = 0; i < input_queues_.size(); i++) {
    // If input queue closed, namely EOE(end of epoch) from
    // dataset reader to here, read failed
    auto queue = input_queues_[i];

    // read LoDTensorArray from queue
    bool success = true;
    auto tensor_arr = queue->Pop(&success);
    if (!success) return false;

    if (tensor_arr.size() == 1) {
      // input array length = 1, treat input type as LoDTensor
      // FIXME(dkp): this may incur error if batch size = 1
      auto tensor = tensor_arr[0];
      if (!tensor.IsInitialized()) return false;

      // get dst variable from scope and check status
      auto name = input_var_names_[i];
      auto* var = scope->Var(name);

      // share input tensor to dst variable
      auto* dst_tensor = var->GetMutable<LoDTensor>();
      dst_tensor->ShareDataWith(tensor);
      dst_tensor->set_lod(tensor.lod());
    } else {
      // input array length > 1 treat input type as LoDTensorArray
      for (auto tensor : tensor_arr) {
        if (!tensor.IsInitialized()) return false;
      }

      // get dst variable from scope and check status
      auto name = input_var_names_[i];
      auto* var = scope->Var(name);

      // share input tensor to dst variable
      auto& dst_tensor_arr = *(var->GetMutable<LoDTensorArray>());
      for (auto& tensor : dst_tensor_arr) tensor.clear();
      dst_tensor_arr.clear();
      dst_tensor_arr.reserve(tensor_arr.size());
      for (size_t i = 0; i < tensor_arr.size(); i++) {
        dst_tensor_arr.emplace_back(tensor_arr[i]);
      }
    }
  }
  return true;
}

void signal_handler(int sig_num) { _exit(-1); }

void MapRunner::StartMapThread(const Scope* scope) {
  map_thread_ = std::thread([this, scope]() -> void {
    // MapThread may crash with SIGSEGV singal in Executor::Prepare
    // when Python program break and exit, catch SIGSEGV singal and
    // exit thread silently
    signal(SIGSEGV, signal_handler);

    auto& scope_ = scope->NewScope();
    framework::Executor executor(place_);
    while (!shutdown_) {
      // check running or shutdown
      std::unique_lock<std::mutex> lock(mutex_);
      running_cond_.wait(lock, [this] { return running_ || shutdown_; });
      if (shutdown_) break;

      // Step 1: get input LoDTensor and share into Scope
      bool success = ShareInputsIntoScope(&scope_);
      if (!success) {
        for (auto& queue : output_queues_) {
          while (queue->Size()) sleep(0.5);
          queue->Close();
        }
        running_ = false;
        continue;
      }

      // Step 2: run ops by executor without fetch
      try {
        executor.Run(*map_block_->Program(), &scope_,
                     static_cast<int>(map_block_->ID()), false, true,
                     output_var_names_, false, true);
      } catch (...) {
        break;
      }

      // Step 3: fetch output variable to LoDTensor vector
      //        and push to output queue
      for (size_t i = 0; i < output_var_names_.size(); i++) {
        auto* out_var = scope_.FindVar(output_var_names_[i]);
        PADDLE_ENFORCE_NOT_NULL(
            out_var, platform::errors::NotFound(
                         "The output variable %s is not found in Map "
                         "program's internal scope",
                         output_var_names_[i]));
        CheckOutputVarStatus(*out_var, output_var_names_[i]);

        if (out_var->IsType<LoDTensor>()) {
          framework::LoDTensorArray t_arr(1);
          copy_tensor(out_var->Get<LoDTensor>(), &t_arr[0]);
          output_queues_[i]->Push(t_arr);
        } else {
          auto out_arr = out_var->Get<LoDTensorArray>();
          framework::LoDTensorArray t_arr(out_arr.size());
          for (size_t i = 0; i < out_arr.size(); i++) {
            copy_tensor(out_arr[i], &t_arr[i]);
          }
          output_queues_[i]->Push(t_arr);
        }
      }
    }
    scope->DeleteScope(&scope_);
  });
}

void MapRunner::CheckOutputVarStatus(const Variable& var,
                                     const std::string& var_name) {
  // only LoDTensor & LoDTensorArray variable type support currently
  if (var.IsType<LoDTensor>()) {
    PADDLE_ENFORCE_EQ(var.Get<LoDTensor>().IsInitialized(), true,
                      platform::errors::InvalidArgument(
                          "The tensor in output variable %s get from Map"
                          "program's internal scope is not initialized.",
                          var_name));
  } else if (var.IsType<LoDTensorArray>()) {
    auto tensor_array = var.Get<LoDTensorArray>();
    for (auto tensor : tensor_array) {
      PADDLE_ENFORCE_EQ(tensor.IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "The tensor in LoDTensorArray of output "
                            "variable %s get from Map program's internal "
                            "scope is not initialized.",
                            var_name));
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "MapOp can only support LoDTensor or LoDTensorArray"));
  }
}

void MapRunner::ShutDown() {
  // close all output queue, op after this op can shutdown itself
  for (auto queue : output_queues_) {
    if (queue && !queue->IsClosed()) queue->Close();
  }

  shutdown_ = true;
  running_ = false;
  running_cond_.notify_all();

  if (map_thread_.joinable()) map_thread_.join();
}

void MapRunner::Reset() {
  for (auto queue : output_queues_) queue->ReOpen();

  running_ = true;
  running_cond_.notify_all();
}

// initialization static variables out of MapRunnerManager
MapRunnerManager* MapRunnerManager::pm_instance_ptr_ = nullptr;
std::mutex MapRunnerManager::m_;

}  // data
}  // namespace operators
}  // namespace paddle
