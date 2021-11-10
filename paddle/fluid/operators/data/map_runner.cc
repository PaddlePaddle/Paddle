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

#include "paddle/fluid/operators/data/map_runner.h"
#include "paddle/fluid/framework/executor_cache.h"

namespace paddle {
namespace operators {
namespace data {

MapRunner::MapRunner(
    const std::shared_ptr<BlockDesc> global_block,
    const platform::Place &place, int64_t start_op_index,
    int64_t end_op_index, int64_t program_id,
    const std::vector<std::string> &input_var_names,
    const std::vector<std::string> &output_var_names,
    const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> input_queues,
    const std::vector<std::shared_ptr<LoDTensorBlockingQueue>> output_queues)
    : thread_pool_(1),
      running_(true),
      global_block_(global_block),
      place_(place),
      start_op_index_(start_op_index),
      end_op_index_(end_op_index),
      program_id_(program_id),
      input_var_names_(input_var_names),
      output_var_names_(output_var_names),
      input_queues_(input_queues),
      output_queues_(output_queues) {

  VLOG(1) << "MapRunner init";

  PADDLE_ENFORCE_GT(end_op_index_, start_op_index_,
                    platform::errors::InvalidArgument(
                        "end_op_index should be greater than start_op_index, "
                        "but recieve %d <= %d.",
                        end_op_index_, start_op_index_));
  PADDLE_ENFORCE_EQ(input_var_names_.size(), input_queues_.size(),
                    platform::errors::InvalidArgument(
                        "input_var_names length should be equal to input_queues length, "
                        "but recieve %d != %d.",
                        input_var_names_.size(),
                        input_var_names_.size()));
  PADDLE_ENFORCE_EQ(output_var_names_.size(), output_queues_.size(),
                    platform::errors::InvalidArgument(
                        "output_var_names length should be equal to output_queues length, "
                        "but recieve %d != %d.",
                        output_var_names_.size(),
                        output_var_names_.size()));

  // Step1: prepare executor
  auto *program = global_block_->Program();
  auto cache_info = framework::GetExecutorInfoFromCache(
      *program, place_, start_op_index_, end_op_index_,
      /*is_grad=*/false, program_id, &scope_);
  auto &parallel_executor = cache_info.first;

  // Step2: parset persistable variables
  auto &skip_eager_delete_vars =
      framework::ExecutorInfoCache::Instance().SkipEagerDeleteVars(
          program_id, /*is_grad=*/false);
  if (cache_info.second /*is_new_created*/) {
    skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                  output_var_names.begin(),
                                  output_var_names.end());
    framework::details::ParseSafeEagerDeletionSkipVars(
        *program, end_op_index, output_var_names, &skip_eager_delete_vars);
  }

  // Step3: start prefetch thread
  StartMapThread(parallel_executor, skip_eager_delete_vars);
}

bool MapRunner::ShareInputsIntoScope() {
  for (size_t i = 0; i < input_queues_.size(); i++) {
    // If input queue closed, namely EOE(end of epoch) from
    // dataset reader to here, read failed
    auto queue = input_queues_[i];
    if (queue->IsClosed()) return false;

    // read LoDTensorArray
    bool success = true;
    auto lod_tensor_arr = queue->Pop(&success);
    if (!success) return false;

    // read LoDTensor
    auto tensor = lod_tensor_arr[0];
    if(!tensor.IsInitialized()) return false; 

    // get input variable from scope and check status
    auto name = input_var_names_[i];
    auto* var = scope_.Var(name);
    if (!var->IsType<LoDTensor>() || !var->IsInitialized()) return false;

    // share input tensor to variable
    auto* dst_tensor = var->GetMutable<LoDTensor>();
    dst_tensor->ShareDataWith(tensor);
    dst_tensor->set_lod(tensor.lod());
  }
  return true;
}

void MapRunner::StartMapThread(std::shared_ptr<ParallelExecutor> executor,
                                   const std::vector<std::string> &skip_vars) {
  thread_pool_.enqueue([this, executor, skip_vars]() -> void {
    while (running_.load()) {
      // Step1: get input LoDTensor and share into Scope
      bool success = ShareInputsIntoScope();
      if (!success) {
        Shutdown();
        break;
      }

      // Step2: run ops by executor without fetch
      executor->RunWithoutFetch(skip_vars);

      // Step3: fetch output variable to LoDTensor vector
      //        and push to output queue
      for (size_t i = 0; i < output_var_names_.size(); i++) {
        framework::LoDTensorArray t_arr(1);
        auto *out_var = scope_.FindVar(output_var_names_[i]);
        PADDLE_ENFORCE_NOT_NULL(
            out_var, platform::errors::NotFound(
                         "The output variable %s is not found in DataLoader "
                         "program's internal scope",
                         output_var_names_[i]));
        CheckOutputVarStatus(*out_var, output_var_names_[i]);
        copy_tensor(out_var->Get<LoDTensor>(), &t_arr[0]);
        output_queues_[i]->Push(t_arr);
      }
    }
  });
}

void MapRunner::CheckOutputVarStatus(const Variable &var,
                                    const std::string &var_name) {
  // only LoDTensor variable type support currently
  PADDLE_ENFORCE_EQ(
      var.IsType<LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "The output variable %s get from DataLoader program's "
          "internal scope holds wrong type. Expect type is "
          "LoDTensor, but receive type is %s.",
          var_name, platform::demangle(framework::ToTypeName(var.Type()))));
  PADDLE_ENFORCE_EQ(var.Get<LoDTensor>().IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "The tensor in output variable %s get from DataLoader "
                        "program's internal scope is not initialized.",
                        var_name));
}

void MapRunner::Shutdown() {
  VLOG(1) << "MapRunner shutdown";
  // close all output queue, op after this op can shutdown itself
  for (auto queue :  output_queues_) {
    queue->Close();
  }

  // set running_ as false to exit map thread, then release thread pool
  running_ = false;
  // FIXME: ThreadPool doesn't have shutdown method
  delete &thread_pool_;
}

// initialization static variables out of MapRunnerManager
MapRunnerManager *MapRunnerManager::pm_instance_ptr_ = nullptr;
std::mutex MapRunnerManager::m_;

}  // data
}  // namespace operators
}  // namespace paddle
