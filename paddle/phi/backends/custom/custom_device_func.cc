#include "paddle/phi/backends/custom/custom_device_func.h"

#if defined(PADDLE_WITH_CUSTOM_DEVICE_SUB_BUILD)
#include "common/custom_device_func.h"
#endif

namespace phi {

std::unique_ptr<CustomDeviceFuncBase> CreateCustomDeviceFunc() {
#if defined(PADDLE_WITH_CUSTOM_DEVICE_SUB_BUILD)
  return std::make_unique<CustomDeviceFunc>();
#else
  return std::make_unique<CustomDeviceFuncBase>();
#endif
}

}  // namespace phi
