core_gather_headers()

set(SRCS
    custom_device_backend_api.cc custom_device_util.cc
    custom_device_intrinsics.cc custom_device_intrinsics_reduce.cc
    custom_device_intrinsics_float16.cc)

# 编译为 CINN 的 Custom Device 运行时库
cc_library(
  cinn_custom_device_runtime
  SRCS ${SRCS}
  DEPS phi_core # 必须保留，否则找不到 DeviceManager 和 Place
)
