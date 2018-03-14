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

#include <unordered_set>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"

#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

struct AllReduceCallBack {
  void operator()(framework::OperatorBase* op);

  std::unordered_set<std::string> param_grad_names_;
  platform::DeviceContext dev_ctx;
};

class ParallelExecutor {
  explicit ParallelExecutor(const std::vector<platform::Place>& places,
                            const std::unordered_set& params);

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDesc& prog, Scope* scope, int block_id,
           bool create_local_scope = true, bool create_vars = true);

 private:
  std::vector<framework::Executor> exes_;
  std::vector<framework::Scope*> scopes_;
  std::vector<AllReduceCallBack> all_reduce_callbacks_;
  platform::Communicator nccl_com_;
};

}  // namespace framework
}  // namespace paddle
