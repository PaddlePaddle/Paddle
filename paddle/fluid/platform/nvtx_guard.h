// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA
#include "gflags/gflags.h"
#include "paddle/fluid/platform/dynload/nvtx.h"
DECLARE_bool(enable_nvtx);
DECLARE_bool(log_nvtx_error);
#endif
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace platform {

class NVTXGuard {
 public:
  static bool IsEnabled() {
#ifdef PADDLE_WITH_CUDA
    return FLAGS_enable_nvtx;
#else
    return false;
#endif
  }

  explicit NVTXGuard(const char* name) {
#ifdef PADDLE_WITH_CUDA
    if (FLAGS_enable_nvtx) {
      auto err_code = platform::dynload::nvtxRangePushA(name);
      if (err_code != 0 && FLAGS_log_nvtx_error) {
        LOG(WARNING) << "NVTXGuard start error: " << name
                     << " , code = " << err_code;
      }
    }
#endif
  }

  explicit NVTXGuard(const std::string& name) : NVTXGuard(name.c_str()) {}

  explicit NVTXGuard(std::string&& name) : NVTXGuard(name.c_str()) {}

  ~NVTXGuard() {
#ifdef PADDLE_WITH_CUDA
    if (FLAGS_enable_nvtx) {
      auto err_code = platform::dynload::nvtxRangePop();
      if (err_code != 0 && FLAGS_log_nvtx_error) {
        LOG(WARNING) << "NVTXGuard exit error, code = " << err_code;
      }
    }
#endif
  }

  DISABLE_COPY_AND_ASSIGN(NVTXGuard);
};

}  // namespace platform
}  // namespace paddle
