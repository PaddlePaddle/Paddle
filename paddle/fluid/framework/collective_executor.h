/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_FLUID_FRAMEWORK_COLLECTIVE_EXECUTOR_H_
#define PADDLE_FLUID_FRAMEWORK_COLLECTIVE_EXECUTOR_H_

#include <string>
#include <vector>
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class NCCLInfo {
 public:
  NCCLInfo() {}
  virtual ~NCCLInfo() {}
 public:
  int local_rank_;
  int global_ranks_;
  int my_global_rank_;
  ncclUniqueId nccl_id_;
  ncclComm_t comm_;
  cudaStream_t stream_;
};


class CollectiveExecutor {
 public:
  explicit CollectiveExecutor(Scope& scope, const platform::Place& place);  // NOLINT
  virtual ~CollectiveExecutor() {}
  void InitNCCL();
  void SetNCCLId(const NCCLInfo& nccl_info);
  NCCLInfo GetNCCLId();
  void SetRankInfo(
      const int local_rank,
      const int global_rank,
      const int ranks);
  void RunStartupProgram(const ProgramDesc& program, Scope* scope);
  void SynchronizeModel(
      const ProgramDesc& program,
      int root_rank,
      Scope* scope);
  std::vector<float> RunFromFile(const ProgramDesc& main_program,
                                 const std::string& data_feed_desc_str,
                                 const std::vector<std::string>& filelist,
                                 const std::vector<std::string>& fetch_names);

 public:
  Scope& root_scope_;
  platform::Place place_;
  NCCLInfo nccl_info_;
};
}  // namespace framework
}  // namespace paddle

#endif  // PADDLE_FLUID_FRAMEWORK_COLLECTIVE_EXECUTOR_H_
