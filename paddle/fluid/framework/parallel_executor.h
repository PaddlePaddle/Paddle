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

#pragma once

#include <atomic>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/execution_strategy.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/details/reference_count_pass.h"
#endif

namespace paddle {
namespace framework {

class ParallelExecutorPrivate;

using details::BuildStrategy;
using details::ExecutionStrategy;

class ParallelExecutor {
  DISABLE_COPY_AND_ASSIGN(ParallelExecutor);

 public:
  explicit ParallelExecutor(const std::vector<platform::Place> &places,
                            const std::unordered_set<std::string> &params,
                            const std::unordered_set<std::string> &bcast_vars,
                            const ProgramDesc &main_program,
                            const std::string &loss_var_name, Scope *scope,
                            const std::vector<Scope *> &local_scopes,
                            const ExecutionStrategy &exec_strategy,
                            const BuildStrategy &build_strategy,
                            size_t num_trainers = 1, size_t trainer_id = 0);

  ~ParallelExecutor();

  std::vector<Scope *> &GetLocalScopes();

  /**
   * Feed tensors to local scopes. The size of tensors should be equal to the
   * size of local scopes.
   */
  void FeedTensorsIntoLocalScopes(
      const std::vector<std::unordered_map<std::string, LoDTensor>> &tensors);

  void FeedAndSplitTensorIntoLocalScopes(
      const std::unordered_map<std::string, LoDTensor> &tensors);

  void Run(const std::vector<std::string> &fetch_tensors,
           const std::string &fetched_var_name);

 private:
  void BCastParamsToDevices(const std::unordered_set<std::string> &vars) const;

  std::unique_ptr<ParallelExecutorPrivate> member_;

#ifdef PADDLE_WITH_CUDA
  // ref_cnts_ is only initialized when ParallelExecutor constructs, and then
  // keeps unchanged
  // Before each iteration, cur_ref_cnts_ is reset to ref_cnts_
  details::DeviceReferenceCountMap ref_cnts_;
  details::AtomicDeviceReferenceCountMap cur_ref_cnts_;
  details::DeviceGarbageCollectorMap gcs_;

  void ResetReferenceCount() {
    for (auto &pair1 : ref_cnts_) {
      for (auto &pair2 : *(pair1.second)) {
        (*(cur_ref_cnts_[pair1.first]))[pair2.first] = pair2.second;
      }
    }
  }
#endif
};

}  // namespace framework
}  // namespace paddle
