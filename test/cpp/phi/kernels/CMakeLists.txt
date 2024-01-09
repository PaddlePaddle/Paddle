cc_test(
  test_math_function
  SRCS test_math_function.cc
  DEPS phi common)
if(WITH_GPU)
  nv_test(
    test_math_function_gpu
    SRCS test_math_function.cu
    DEPS phi common)
  nv_test(
    test_broadcast_gpu
    SRCS test_ternary_broadcast.cu
    DEPS gtest)
endif()
if(WITH_ROCM)
  hip_test(
    test_math_function_gpu
    SRCS test_math_function.cu
    DEPS phi common)
endif()

cc_test(
  test_cpu_vec
  SRCS test_cpu_vec.cc
  DEPS phi common)

# For String Kernels
cc_test(
  test_strings_lower_upper_dev_api
  SRCS test_strings_lower_upper_dev_api.cc
  DEPS phi common)
if(WITH_GPU)
  nv_test(
    test_strings_lower_upper_dev_gpu_api
    SRCS test_strings_lower_upper_dev_api.cu
    DEPS phi common)
elseif(WITH_ROCM)
  hip_test(
    test_strings_lower_upper_dev_gpu_api
    SRCS test_strings_lower_upper_dev_api.cu
    DEPS phi common)
endif()

cc_test(
  test_strings_copy_dev_api
  SRCS test_strings_copy_dev_api.cc
  DEPS phi common)
if(WITH_GPU)
  nv_test(
    test_strings_copy_dev_gpu_api
    SRCS test_strings_copy_dev_api.cu
    DEPS phi common)
elseif(WITH_ROCM)
  hip_test(
    test_strings_copy_dev_gpu_api
    SRCS test_strings_copy_dev_api.cu
    DEPS phi common)
endif()

cc_test(
  test_memcpy_dev_api
  SRCS test_memcpy_dev_api.cc
  DEPS phi common)

cc_test(
  test_transfer_layout_dev_api
  SRCS test_transfer_layout_dev_api.cc
  DEPS phi common)

if(WITH_GPU)
  nv_test(
    test_gpu_timer
    SRCS test_gpu_timer.cu
    DEPS gtest)
  nv_test(
    test_auto_tune
    SRCS test_auto_tune.cu
    DEPS gtest)
  cc_test(
    test_fused_adam_kernel
    SRCS test_fused_adam_kernel.cc
    DEPS gtest phi common)
elseif(WITH_ROCM)
  hip_test(
    test_gpu_timer
    SRCS test_gpu_timer.cu
    DEPS gtest)
  hip_test(
    test_auto_tune
    SRCS test_auto_tune.cu
    DEPS gtest)
endif()

cc_test(
  test_cache
  SRCS test_cache.cc
  DEPS gtest phi common)

cc_test(
  strided_memcpy_test
  SRCS strided_memcpy_test.cc
  DEPS phi common)

cc_test(
  sequence_padding_test
  SRCS sequence_padding_test.cc
  DEPS phi common)

cc_test(
  sequence_pooling_test
  SRCS sequence_pooling_test.cc
  DEPS phi common)
