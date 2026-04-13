core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  flags.cc
  intrinsic.cc
  cinn_runtime.cc
  intrinsic_types.cc
  backend_api.cc
  custom_device/custom_device_backend_api.cc
  custom_device/custom_device_util.cc
  custom_device/custom_device_intrinsics.cc
  custom_device/custom_device_intrinsics_reduce.cc
  custom_device/custom_device_intrinsics_float16.cc)

cinn_cc_library(
  cinn_runtime SRCS cinn_runtime.cc buffer.cc #cinn_x86_device_impl.cc
)

if(WITH_OPENMP)
  cinn_cc_library(tiny_runtime STATIC SRCS tiny_runtime.cc)
endif()

add_subdirectory(cuda)
add_subdirectory(cpu)
if(WITH_ROCM)
  add_subdirectory(hip)
endif()

if(WITH_SYCL)
  add_subdirectory(sycl)
endif()

if(WITH_CUSTOM_DEVICE)
  add_subdirectory(custom_device)
endif()
