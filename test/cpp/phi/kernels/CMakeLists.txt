paddle_test(test_math_function SRCS test_math_function.cc)
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

paddle_test(test_cpu_vec SRCS test_cpu_vec.cc)

# For String Kernels
paddle_test(test_strings_lower_upper_dev_api SRCS
            test_strings_lower_upper_dev_api.cc)
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

paddle_test(test_strings_copy_dev_api SRCS test_strings_copy_dev_api.cc)
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

paddle_test(test_memcpy_dev_api SRCS test_memcpy_dev_api.cc)

paddle_test(test_transfer_layout_dev_api SRCS test_transfer_layout_dev_api.cc)

if(WITH_GPU)
  nv_test(
    test_gpu_timer
    SRCS test_gpu_timer.cu
    DEPS gtest)
  nv_test(
    test_auto_tune
    SRCS test_auto_tune.cu
    DEPS gtest)
  paddle_test(test_fused_adam_kernel SRCS test_fused_adam_kernel.cc)
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

paddle_test(test_cache SRCS test_cache.cc)

paddle_test(strided_memcpy_test SRCS strided_memcpy_test.cc)

paddle_test(sequence_padding_test SRCS sequence_padding_test.cc)

paddle_test(sequence_pooling_test SRCS sequence_pooling_test.cc)
