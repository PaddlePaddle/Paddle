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
                   const std::vector<std::string> &output_var_names)
    : running_(true),
      global_block_(global_block),
      place_(place),
      start_op_index_(start_op_index),
      end_op_index_(end_op_index),
      program_id_(program_id),
      output_var_names_(output_var_names) {
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
  parallel_executor->RunWithoutFetch(skip_eager_delete_vars);
}

void Pipeline::CheckOutputVarStatus(const Variable &var,
                                    const std::string &var_name) {
  // only LoDTensor variable type support currently
  PADDLE_ENFORCE_EQ(var.IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "The tensor in output variable %s get from DataLoader "
                        "program's internal scope is not initialized.",
                        var_name));
  PADDLE_ENFORCE_EQ(
      var.IsType<LoDTensor>(), true,
      platform::errors::InvalidArgument(
          "The output variable %s get from DataLoader program's "
          "internal scope holds wrong type. Expect type is "
          "LoDTensor, but receive type is %s.",
          var_name, platform::demangle(framework::ToTypeName(var.Type()))));
}

void Pipeline::ReadNext(std::vector<Variable *> &out_vars) {
  PADDLE_ENFORCE_EQ(out_vars.size(), output_var_names_.size(), 
        platform::errors::InvalidArgument(
          "Out variable number should equal to output variable name "
          "number, but receive %d != %d", out_vars.size(),
          output_var_names_.size()));
  for (size_t i = 0; i < output_var_names_.size(); i++) {
    auto *out_var = scope_.FindVar(output_var_names_[i]);
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "The output variable %s is not found in DataLoader "
                     "program's internal scope",
                     output_var_names_[i]));
    auto out_queue = out_var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
    if (out_queue->IsClosed()) {
      running_.store(false);
      return;
    }

    bool success = true;
    auto outputs = out_queue->Pop(&success);
    PADDLE_ENFORCE_EQ(success, true, 
        platform::errors::PreconditionNotMet("Read from output queue %s failed", output_var_names_[i]));
    
    // CheckOutputVarStatus(*(out_vars[i]), output_var_names_[i]);
    copy_tensor(outputs.at(0), out_vars[i]->GetMutable<LoDTensor>());
    for (auto &output: outputs) output.clear();
    outputs.clear();
  }
}

// initialization static variables out of PipelineManager
PipelineManager *PipelineManager::pm_instance_ptr_ = nullptr;
std::mutex PipelineManager::m_;

}  // data
}  // namespace operators
}  // namespace paddle
