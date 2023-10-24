add_subdirectory(benchmark)
if(WITH_CINN)
  add_subdirectory(cinn)
endif()
add_subdirectory(controlflow)
add_subdirectory(detection)
if(WITH_DLNNE)
  add_subdirectory(dlnne)
endif()
add_subdirectory(elementwise)
add_subdirectory(fused)
if(WITH_LITE)
  add_subdirectory(lite)
endif()
add_subdirectory(math)
if(WITH_MKLDNN)
  add_subdirectory(mkldnn)
endif()
add_subdirectory(nccl)
if(WITH_PSCORE)
  add_subdirectory(pscore)
endif()
add_subdirectory(reader)
add_subdirectory(reduce_ops)
# TODO(gouzil): enable this after the bug is fixed. windows: Exit code 0xc000007b, pr: #53470
# if(WITH_GPU AND TENSORRT_FOUND)
#   add_subdirectory(tensorrt)
# endif()

set(COMMON_OP_DEPS ${COMMON_OP_DEPS} executor)

if(WITH_XPU)
  cc_test(
    beam_search_decode_op_xpu_test
    SRCS beam_search_decode_op_xpu_test.cc
    DEPS lod_tensor)
  set(COMMON_OP_DEPS ${COMMON_OP_DEPS} xpulib)
endif()

cc_test(
  test_common_infer_shape_functions
  SRCS test_common_infer_shape_functions.cc
  DEPS common_infer_shape_functions
       ${COMMON_OP_DEPS}
       activation_op
       generated_op
       elementwise_add_op
       phi
       generated_static_op)
cc_test(
  gather_test
  SRCS gather_test.cc
  DEPS tensor)
cc_test(
  assign_op_test
  SRCS assign_op_test.cc
  DEPS generated_static_op)
cc_test(
  scatter_test
  SRCS scatter_test.cc
  DEPS tensor phi)
cc_test(
  beam_search_decode_op_test
  SRCS beam_search_decode_op_test.cc
  DEPS lod_tensor)
cc_test(
  save_load_op_test
  SRCS save_load_op_test.cc
  DEPS save_op load_op)
cc_test(
  save_load_combine_op_test
  SRCS save_load_combine_op_test.cc
  DEPS save_combine_op load_combine_op)
if(WITH_CINN)
  set(CINN_DEPS cinn_compiler python)
endif()
if(WITH_GPU)
  nv_test(
    dropout_op_test
    SRCS dropout_op_test.cc
    DEPS dropout_op tensor phi)
  nv_test(
    test_leaky_relu_grad_grad_functor
    SRCS test_leaky_relu_grad_grad_functor.cc
         test_leaky_relu_grad_grad_functor.cu
    DEPS tensor device_context eigen3)
  nv_test(
    feed_forward_test
    SRCS feed_forward_test.cu
    DEPS fleet_executor elementwise_add_op matmul_op tensor phi ${CINN_DEPS})
elseif(WITH_ROCM)
  hip_test(
    dropout_op_test
    SRCS dropout_op_test.cc
    DEPS dropout_op tensor phi)
  hip_test(
    test_leaky_relu_grad_grad_functor
    SRCS test_leaky_relu_grad_grad_functor.cc
         test_leaky_relu_grad_grad_functor.cu
    DEPS tensor device_context eigen3)
else()
  cc_test(
    test_leaky_relu_grad_grad_functor
    SRCS test_leaky_relu_grad_grad_functor.cc
    DEPS tensor device_context eigen3)
endif()
cc_test(
  share_buffer_op_cpp_test
  SRCS share_buffer_op_test.cc
  DEPS lod_tensor device_context generated_static_op)

if(WITH_CINN)
  cc_test_old(
    op_debug_string_test
    SRCS
    op_debug_string_test.cc
    DEPS
    executor
    fleet_executor
    cinn_compiler
    recurrent_op_helper
    recurrent_op
    elementwise_add_op
    ${COMMON_OP_DEPS}
    python)
else()
  paddle_test(op_debug_string_test SRCS op_debug_string_test.cc)
endif()

if(WITH_GPU)
  cc_test(
    copy_cross_scope_test
    SRCS copy_cross_scope_test.cc
    DEPS op_registry copy_cross_scope_op scope device_context enforce executor)
endif()

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(op_debug_string_test)
endif()
