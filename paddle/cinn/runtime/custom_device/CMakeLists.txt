core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  custom_device_util.cc
  custom_device_backend_api.cc
  custom_device_module.cc
  custom_device_intrinsics.cc
  custom_device_intrinsics_reduce.cc
  custom_device_intrinsics_float16.cc)

target_link_libraries(cinn_custom_device_runtime PUBLIC phi_core)
# 或者对应的 phi 库名，确保能找到 custom_device.h
