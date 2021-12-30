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

#include "paddle/fluid/operators/data/pipeline.h"
#include "paddle/fluid/framework/executor_cache.h"

namespace paddle {
namespace operators {
namespace data {

Pipeline::Pipeline(const std::shared_ptr<BlockDesc> global_block,
                   const platform::Place &place, int64_t start_op_index,
                   int64_t end_op_index, int64_t program_id,
                   const std::vector<std::string> &output_var_names,
                   size_t prefetch_queue_size)
    : thread_pool_(1),
      closed_(false),
      global_block_(global_block),
      place_(place),
      start_op_index_(start_op_index),
      end_op_index_(end_op_index),
      program_id_(program_id),
      output_var_names_(output_var_names),
      prefetch_queue_size_(prefetch_queue_size),
      prefetch_queue_(prefetch_queue_size) {
  VLOG(1) << "Pipeline init";

  PADDLE_ENFORCE_GT(end_op_index_, start_op_index_,
                    platform::errors::InvalidArgument(
                        "end_op_index should be greater than start_op_index, "
                        "but recieve %d <= %d.",
                        end_op_index_, start_op_index_));

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
    // DataLoader program do not has input variables, not need to
    // skip memory reuse for input variables here
    skip_eager_delete_vars.insert(skip_eager_delete_vars.end(),
                                  output_var_names.begin(),
                                  output_var_names.end());
    framework::details::ParseSafeEagerDeletionSkipVars(
        *program, end_op_index, output_var_names, &skip_eager_delete_vars);
  }

  // Step3: start prefetch thread
  StartPrefetchThread(parallel_executor, skip_eager_delete_vars);
}

void Pipeline::StartPrefetchThread(std::shared_ptr<ParallelExecutor> executor,
                                   const std::vector<std::string> &skip_vars) {
  thread_pool_.enqueue([this, executor, skip_vars]() -> void {
    while (!closed_.load()) {
      LOG(ERROR) << "Executor run a iter start";
      // Step1: run ops by executor without fetch
      executor->RunWithoutFetch(skip_vars);

      // Step2: fetch output variable to LoDTensor vector
      framework::LoDTensorArray t_arr;
      t_arr.resize(output_var_names_.size());
      for (size_t i = 0; i < output_var_names_.size(); i++) {
        auto *out_var = scope_.FindVar(output_var_names_[i]);
        PADDLE_ENFORCE_NOT_NULL(
            out_var, platform::errors::NotFound(
                         "The output variable %s is not found in DataLoader "
                         "program's internal scope",
                         output_var_names_[i]));
        // CheckOutputVarStatus(*out_var, output_var_names_[i]);
        // copy_tensor(out_var->Get<LoDTensor>(), &t_arr[i]);
        auto out_queue = out_var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
        bool success = true;
        auto outputs = out_queue->Pop(&success);
        PADDLE_ENFORCE_EQ(success, true, 
            platform::errors::PreconditionNotMet("Read from input queue failed"));
        copy_tensor(outputs.at(0), &t_arr[i]);
      }

      // TODO: dataset drain check
      // if dataset drained:
      //     closed_.store(true)
      //     break

      // Step3: put LoDTensorArray to prefetch blocking_queue
      prefetch_queue_.Push(t_arr);
      LOG(ERROR) << "Executor run a iter finish";
    }
  });
}

void Pipeline::CheckOutputVarStatus(const Variable &var,
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

void Pipeline::ReadNext(std::vector<Variable *> &out_vars) {
  bool ok = true;
  auto vars = prefetch_queue_.Pop(&ok);
  PADDLE_ENFORCE_EQ(
      ok, true, platform::errors::Unavailable("Pop prefetch queue failed."));
  PADDLE_ENFORCE_EQ(
      out_vars.size(), vars.size(),
      platform::errors::InvalidArgument(
          "Output variable number to read should be variable number "
          "read from prefetch queue, but recieved %d != %d",
          out_vars.size(), output_var_names_.size()));

  for (size_t i = 0; i < vars.size(); i++) {
    copy_tensor(vars[i], out_vars[i]->GetMutable<LoDTensor>());
  }
}

void Pipeline::ShutDown() {
  VLOG(1) << "Pipeline close";
  closed_.store(true);
  prefetch_queue_.Close();
}

void Pipeline::Reset() {
  // (TODO)Step1: reset dataset
  //
  // Step2: reopen pipeline
  prefetch_queue_.ReOpen();
  closed_.store(false);
  // StartPrefetchThread();
}

// initialization static variables out of PipelineManager
PipelineManager *PipelineManager::pm_instance_ptr_ = nullptr;
std::mutex PipelineManager::m_;

}  // data
}  // namespace operators
}  // namespace paddle
