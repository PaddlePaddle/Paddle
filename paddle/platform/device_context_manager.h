#include "paddle/platform/device_context.h"

#define kNUM_GPUS = 16;
#define kNUM_STREAMS = 16;

namespace paddle {
namespace platform {

class DeviceContextManager {
 public:
  DeviceContextManager();
  ~DeviceContextManager();

  DeviceContext* GetDeviceContext(Place& place);
  DeviceContext* GetIODeviceContext(Place& place);

 private:
#ifndef PADDLE_ONLY_CPU
  std::array<std::array<CUDADeviceContext*, kNUM_STREAMS>, kNUM_GPUS>
      cuda_contexts_;
  std::array<CDUADeviceContext*, kNUM_GPUS> cuda_io_contexts_;
  std::array<int, kNUM_GPUS> gpu_cnt_;
#endif
};
}
}
