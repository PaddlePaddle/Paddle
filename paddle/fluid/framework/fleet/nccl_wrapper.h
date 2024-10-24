/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#ifdef PADDLE_WITH_NCCL
#include "paddle/phi/backends/dynload/nccl.h"
#endif
#ifdef PADDLE_WITH_RCCL
#include "paddle/phi/backends/dynload/rccl.h"
#endif
#include "paddle/common/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

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
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  ncclUniqueId nccl_id_;
  ncclComm_t comm_;
  gpuStream_t stream_;
#endif
};

class NCCLWrapper {
 public:
  virtual ~NCCLWrapper() {}
  NCCLWrapper() {}

  void InitNCCL();
  void SetNCCLId(const NCCLInfo& nccl_info);
  NCCLInfo GetNCCLId();
  void SetRankInfo(const int local_rank,
                   const int global_rank,
                   const int ranks);
  void SyncVar(const int root_rank,
               const Scope& scope,
               const std::vector<std::string>& var_names);

  static std::shared_ptr<NCCLWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::NCCLWrapper());
    }
    return s_instance_;
  }

 public:
  NCCLInfo nccl_info_;

 private:
  static std::shared_ptr<NCCLWrapper> s_instance_;

 protected:
  static bool is_initialized_;
  DISABLE_COPY_AND_ASSIGN(NCCLWrapper);
};

}  // namespace framework
}  // namespace paddle
