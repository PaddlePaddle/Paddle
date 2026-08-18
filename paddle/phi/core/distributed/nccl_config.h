// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <memory>
#if defined(PADDLE_WITH_RCCL)
#include "paddle/phi/backends/dynload/rccl.h"
#else
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
namespace distributed {

class NCCLConfig {
 public:
  static std::shared_ptr<NCCLConfig> CreateNCCLConfig(
      const std::string& commName = "",
      const int ll_buffsize = -1,
      const int ll128_buffsize = -1,
      const int simple_buffsize = -1,
      const int buffsize_align = -1,
      const int nchannels = -1,
      const std::string& algoStr = "",
      const std::string& protoStr = "",
      const int cta_policy = -1);
  ncclConfig_t* GetOrigin();
  ncclMemOptConfig_t* GetMemOpt();

  NCCLConfig(const std::string& commName,
             const int ll_buffsize,
             const int ll128_buffsize,
             const int simple_buffsize,
             const int buffsize_align,
             const int nchannels,
             const std::string& algoStr,
             const std::string& protoStr,
             const int cta_policy = -1);
  ~NCCLConfig();

 private:
  const std::string commName_;
  const int ll_buffsize_;
  const int ll128_buffsize_;
  const int simple_buffsize_;
  const int buffsize_align_;
  const int nchannels_;
  const std::string algoStr_;
  const std::string protoStr_;
  // One of NCCL_CTA_POLICY_DEFAULT / _EFFICIENCY / _ZERO, or -1 to leave the
  // decision to NCCL. _ZERO selects the zero-SM communication path, which only
  // takes effect for buffers registered as symmetric memory windows.
  const int cta_policy_;

  ncclMemOptConfig_t* nccl_memopt_config_ptr{nullptr};

#if NCCL_VERSION_CODE >= 21400
  ncclConfig_t nccl_config_ = NCCL_CONFIG_INITIALIZER;
  // GetOrigin() only hands a non-null config to ncclCommInitRankConfig* when at
  // least one field was explicitly requested, so the default code path keeps
  // its previous behaviour of passing nullptr.
  bool has_origin_config_{false};
#endif
};

}  // namespace distributed
}  // namespace phi
