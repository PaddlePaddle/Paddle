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

#include "paddle/phi/core/distributed/nccl_config.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/nccl_tools.h"
#include "paddle/phi/core/enforce.h"

namespace phi::distributed {

std::shared_ptr<NCCLConfig> NCCLConfig::CreateNCCLConfig(
    const std::string& commName,
    const int ll_buffsize,
    const int ll128_buffsize,
    const int simple_buffsize,
    const int buffsize_align,
    const int nchannels,
    const std::string& algoStr,
    const std::string& protoStr,
    const int cta_policy) {
  return std::make_shared<NCCLConfig>(commName,
                                      ll_buffsize,
                                      ll128_buffsize,
                                      simple_buffsize,
                                      buffsize_align,
                                      nchannels,
                                      algoStr,
                                      protoStr,
                                      cta_policy);
}

NCCLConfig::NCCLConfig(const std::string& commName,
                       const int ll_buffsize,
                       const int ll128_buffsize,
                       const int simple_buffsize,
                       const int buffsize_align,
                       const int nchannels,
                       const std::string& algoStr,
                       const std::string& protoStr,
                       const int cta_policy)
    : commName_(commName),
      ll_buffsize_(ll_buffsize),
      ll128_buffsize_(ll128_buffsize),
      simple_buffsize_(simple_buffsize),
      buffsize_align_(buffsize_align),
      nchannels_(nchannels),
      algoStr_(algoStr),
      protoStr_(protoStr),
      cta_policy_(cta_policy),
      nccl_memopt_config_ptr(nullptr) {
  if (phi::dynload::ncclCommGenMemOptConfig.IsValid()) {
    nccl_memopt_config_ptr = phi::dynload::ncclCommGenMemOptConfig(
        commName_.empty() ? nullptr : commName_.c_str(),
        ll_buffsize_ >= 0 ? ll_buffsize_ : -1,
        ll128_buffsize_ >= 0 ? ll128_buffsize_ : -1,
        simple_buffsize_ >= 0 ? simple_buffsize_ : -1,
        buffsize_align_ >= 0 ? buffsize_align_ : -1,
        nchannels_ >= 0 ? nchannels_ : -1,
        algoStr_.empty() ? nullptr : algoStr_.c_str(),
        protoStr_.empty() ? nullptr : protoStr_.c_str());
  }
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  if (cta_policy_ >= 0) {
    // An older library drops a config field it does not know silently, so check
    // the loaded version rather than pretending the request took effect.
    int runtime_version = 0;
    NCCL_CHECK(phi::dynload::ncclGetVersion(&runtime_version));
    PADDLE_ENFORCE_GE(runtime_version,
                      23007,
                      common::errors::Unavailable(
                          "cta_policy needs NCCL 2.30.7 or newer, but the "
                          "loaded library reports version %d.",
                          runtime_version));
    nccl_config_.CTAPolicy = cta_policy_;
    has_origin_config_ = true;
    VLOG(3) << "NCCLConfig " << commName_ << " requests CTAPolicy "
            << cta_policy_;
  }
#else
  PADDLE_ENFORCE_LT(cta_policy_,
                    0,
                    common::errors::Unavailable(
                        "cta_policy needs Paddle to be compiled against NCCL "
                        "2.30.7 or newer, but NCCL_VERSION_CODE is %d.",
                        NCCL_VERSION_CODE));
#endif
}

ncclConfig_t* NCCLConfig::GetOrigin() {
#if defined(PADDLE_WITH_NCCL) && NCCL_VERSION_CODE >= 23007
  if (has_origin_config_) {
    return &nccl_config_;
  }
#endif
  return nullptr;
}

ncclMemOptConfig_t* NCCLConfig::GetMemOpt() { return nccl_memopt_config_ptr; }

NCCLConfig::~NCCLConfig() {
  if (phi::dynload::ncclCommFreeMemOptConfig.IsValid() &&
      nccl_memopt_config_ptr != nullptr) {
    phi::dynload::ncclCommFreeMemOptConfig(nccl_memopt_config_ptr);
  }
}

}  // namespace phi::distributed
