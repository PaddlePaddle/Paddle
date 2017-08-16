#include "paddle/platform/device_context_manager.h"

namespace paddle {
namespace platform {

DeviceContextManager::DeviceContextManager() {
#ifndef PADDLE_ONLY_CPU
  for (size_t i = 0; i < kNUM_GPUS; i++) {
    gpu_cnt_.at(i) = -1;
  }

#endif
}

DeviceContext& DeviceContextManager::GetDeviceContext(Place& place) {
  if (is_cpu_place(place)) {
    return CPUDeviceContext();
  } else
    (is_gpu_place(place)) {
#ifndef PADDLE_ONLY_CPU
      PADDLE_ENFORCE(place.device < kNUM_GPUS,
                     "GPU device id must less than kNUM_GPUS");
      PADDLE_ENFORCE(SetDeviceId(place.device));
      for (auto&& ctx : cuda_contexts_) {
        ctx = new CUDADeviceContext(place);
      }
      gpu_cnt_.at(place.device) = 0;
      return *cuda_contexts_[place.device][0];

#else
      PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
    }
}

DeviceContextManager::~DeviceContextManager() {
#ifndef PADDLE_ONLY_CPU
  for (size_t i = 0; i < kNUM_GPUS; i++) {
    if (gpu_cnt_.at(i) != -1) {
      for (auto&& ctx : cuda_contexts_.at(i)) {
        delete ctx;
      }
      gpu_cnt_.at(i) = -1;
    }
  }

#endif
}
}
}