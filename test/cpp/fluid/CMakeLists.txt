add_subdirectory(memory)
add_subdirectory(benchmark)
add_subdirectory(framework)
add_subdirectory(platform)

add_subdirectory(controlflow)

add_subdirectory(elementwise)
add_subdirectory(fused)
add_subdirectory(math)
if(WITH_ONEDNN)
  add_subdirectory(mkldnn)
endif()
if(WITH_PSCORE)
  add_subdirectory(pscore)
endif()
add_subdirectory(reader)
add_subdirectory(reduce_ops)
if(WITH_GPU AND TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()

set(COMMON_OP_DEPS ${COMMON_OP_DEPS} executor)

paddle_test(gather_test SRCS gather_test.cc)
paddle_test(assign_op_test SRCS assign_op_test.cc)
paddle_test(scatter_test SRCS scatter_test.cc DEPS common)
paddle_test(beam_search_decode_op_test SRCS beam_search_decode_op_test.cc)
paddle_test(save_load_op_test SRCS save_load_op_test.cc)
if(WITH_XPU)
  paddle_test(save_load_op_test_xpu SRCS save_load_op_test_xpu.cc)
  paddle_test(beam_search_op_test_xpu SRCS beam_search_op_test_xpu.cc)
  paddle_test(save_load_combine_op_test_xpu SRCS
              save_load_combine_op_test_xpu.cc)
endif()
paddle_test(save_load_combine_op_test SRCS save_load_combine_op_test.cc)
if(WITH_CINN)
  set(CINN_DEPS python)
endif()
if(WITH_GPU)
  nv_test(
    dropout_op_test
    SRCS dropout_op_test.cc
    DEPS dropout_op tensor phi common global_utils)
  nv_test(
    test_leaky_relu_grad_grad_functor
    SRCS test_leaky_relu_grad_grad_functor.cc
         test_leaky_relu_grad_grad_functor.cu
    DEPS tensor device_context eigen3)
elseif(WITH_ROCM)
  hip_test(
    dropout_op_test
    SRCS dropout_op_test.cc
    DEPS dropout_op tensor phi common)
  hip_test(
    test_leaky_relu_grad_grad_functor
    SRCS test_leaky_relu_grad_grad_functor.cc
         test_leaky_relu_grad_grad_functor.cu
    DEPS tensor device_context eigen3)
else()
  paddle_test(test_leaky_relu_grad_grad_functor SRCS
              test_leaky_relu_grad_grad_functor.cc)
endif()
paddle_test(share_buffer_op_cpp_test SRCS share_buffer_op_test.cc)

if(WITH_CINN)
  paddle_test(op_debug_string_test SRCS op_debug_string_test.cc)
else()
  paddle_test(op_debug_string_test SRCS op_debug_string_test.cc)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(op_debug_string_test)
endif()
