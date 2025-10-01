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

#include "paddle/phi/core/platform/cuda_device_guard.h"
#include "glog/logging.h"

namespace paddle {
namespace platform {

CUDADeviceGuard::~CUDADeviceGuard() {
  static thread_local bool is_first_time_ = true;
  if (prev_id_ != -1) {
    // Do not set device back for the first time, since
    // `cudaGetDevice` returns 0 when `cudaSetDevice` is
    // not called.
    // In that case, if CUDADeviceGuard(7) is called,
    // prev_id will be 0 and we don`t need to set it back to 0.
    // If cudaSetDevice(0) is called, it may use hundreds MB of
    // the gpu memory.
    VLOG(10) << __func__ << " prev_id: " << prev_id_ << ", is_first_time_"
             << is_first_time_;
    if (!(is_first_time_ && prev_id_ == 0)) {
      phi::backends::gpu::SetDeviceId(prev_id_);
      is_first_time_ = false;
    }
  }
}

}  // namespace platform
}  // namespace paddle
