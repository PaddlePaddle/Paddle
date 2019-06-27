if (NOT LITE_WITH_OPENCL)
    return ()
endif()

set(cl_kernel_deps op_params_lite cl_caller cl_engine cl_context cl_wrapper)

lite_cc_library(elementwise_add_opencl SRCS elementwise_add_compute.cc DEPS ${cl_kernel_deps})
lite_cc_library(pool_opencl SRCS pool_compute.cc DEPS ${cl_kernel_deps})

lite_cc_test(test_elementwise_add_opencl SRCS elementwise_add_compute_test.cc DEPS elementwise_add_opencl
  op_registry_lite program_lite
  context_lite
  ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
  )

lite_cc_test(test_pool_opencl SRCS pool_compute_test.cc DEPS pool_opencl
  op_registry_lite program_lite
  context_lite
  ARGS --cl_path=${CMAKE_SOURCE_DIR}/paddle/fluid/lite/opencl
  )

set(opencl_kernels
  elementwise_add_opencl
  pool_opencl
  CACHE INTERNAL "opencl_kernels")
