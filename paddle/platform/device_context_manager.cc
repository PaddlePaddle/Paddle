/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/platform/device_context_manager.h"

namespace paddle {
namespace platform {

DeviceContextManager::DeviceContextManager() {
#ifndef PADDLE_ONLY_CPU
  device_count_ = GetDeviceCount();
  cuda_contexts_.reserve(device_count_);
  for (int i = 0; i < device_count_; i++) {
    cuda_contexts_[i].reserve(kNUM_STREAMS);
  }
#endif
}

DeviceContext& DeviceContextManager::GetDeviceContext(Place& place,
                                                      int stream_id) {
  if (is_cpu_place(place)) {
    if (!cpu_context_) {
      cpu_context_ = new CPUDeviceContext();
    }
    return *cpu_context_;
  } else {
#ifndef PADDLE_ONLY_CPU
    auto gpu_place = boost::get<GPUPlace>(place);
    auto gpu_id = gpu_place.device;
    PADDLE_ENFORCE(gpu_id < device_count_,
                   "GPU device id must less than device count");
    SetDeviceId(gpu_id);
    auto* ctx = cuda_contexts_[gpu_id][stream_id];
    if (!ctx) {
      ctx = new CUDADeviceContext(gpu_place);
    }
    return *ctx;
#else
    PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
  }
}

DeviceContextManager::~DeviceContextManager() {
  delete cpu_context_;
#ifndef PADDLE_ONLY_CPU
  for (int i = 0; i < device_count_; i++) {
    for (auto& ctx : cuda_contexts_[i]) {
      if (ctx) {
        delete ctx;
      }
    }
  }
#endif

  DeviceContextManager* Get() {
    static std::unique_ptr<DeviceContextManager> inst;
    return inst.get();
  }
}
}  // namespace platform
}  // namespace paddle
