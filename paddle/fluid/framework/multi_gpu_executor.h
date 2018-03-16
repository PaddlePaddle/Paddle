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

struct NCCLContext {
  std::vector<platform::CUDADeviceContext*> ctxs_;
  std::vector<ncclComm_t> comms_;

  explicit NCCLContext(const std::vector<platform::Place>& places) {
    std::vector<int> devs;
    devs.reserve(places.size());
    for (auto& p : places) {
      devs.push_back(boost::get<platform::CUDAPlace>(p).device);
      ctxs_.push_back(
          new platform::CUDADeviceContext(boost::get<platform::CUDAPlace>(p)));
    }
    comms_.reserve(places.size());
    platform::dynload::ncclCommInitAll(
        &comms_[0], static_cast<int>(places.size()), &devs[0]);
  }
};

class ExecutorWithAllReduce : public Executor {
 public:
  explicit ExecutorWithAllReduce(const platform::Place& p,
                                 std::unordered_set<std::string>* param_grads,
                                 NCCLContext* nccl_context);

 private:
  void RunOperators(const ExecutorPrepareContext* ctx,
                    const Scope* local_scope) const override;
  platform::CUDADeviceContext* io_ctx_;
  ncclComm_t* comm_;
  std::unordered_set<std::string>* param_grads_;
};

class MultiGPUExecutor {
 public:
  explicit MultiGPUExecutor(const std::vector<platform::Place>& places,
                            const std::unordered_set<std::string>& params);

  /* @Brief
   * Runtime evaluation of the given ProgramDesc under certain Scope
   *
   * @param
   *  ProgramDesc
   *  Scope
   */
  void Run(const ProgramDesc& prog, int block_id,
           bool create_local_scope = true, bool create_vars = true);

 private:
  std::vector<framework::ExecutorWithAllReduce> exes_;
  std::vector<framework::Scope*> scopes_;
  NCCLContext nccl_ctx_;
  std::unordered_set<std::string> param_grads_;
};

}  // namespace framework
}  // namespace paddle
