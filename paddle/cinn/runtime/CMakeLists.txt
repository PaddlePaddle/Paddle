core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  flags.cc
  intrinsic.cc
  cinn_runtime.cc
  intrinsic_types.cc
  custom_function.cc)

cc_library(cinn_runtime SRCS cinn_runtime.cc buffer.cc #cinn_x86_device_impl.cc
)

cc_test(
  test_cinn_runtime
  SRCS cinn_runtime_test.cc
  DEPS cinn_runtime)
cc_test(
  test_custom_function
  SRCS custom_function_test.cc
  DEPS cinncore)

if(WITH_OPENMP)
  cc_library(tiny_runtime STATIC SRCS tiny_runtime.cc)
endif()

add_subdirectory(cuda)
add_subdirectory(cpu)
