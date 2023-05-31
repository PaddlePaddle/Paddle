core_gather_headers()

gather_srcs(
  cinnapi_src
  SRCS
  shared.cc
  cinn_value.cc
  type.cc
  target.cc
  object.cc
  debug_manager.cc
  info_registry.cc
  graph_utils.cc
  context.cc
  axis.cc
  ir_util.cc
  test_helper.cc
  # cuda_test_helper.cc
  arithmatic.cc
  cas.cc
  union_find.cc
  python_interpreter_guard.cc)

message(STATUS "srcs: ${cinnapi_src}")

cc_test(
  test_cinn_value
  SRCS cinn_value_test.cc
  DEPS cinncore)
cc_test(
  test_shared
  SRCS shared_test.cc
  DEPS cinncore)
cc_test(
  test_graph_utils
  SRCS graph_utils_test.cc
  DEPS cinncore)
cc_test(
  test_arithmatic
  SRCS arithmatic_test.cc
  DEPS cinncore)
cc_test(
  test_cas
  SRCS cas_test.cc
  DEPS cinncore)
cc_test(
  test_type
  SRCS type_test.cc
  DEPS cinncore)
cc_test(
  test_axis
  SRCS axis_test.cc
  DEPS cinncore)

cc_test(
  test_fp16_bf16_host
  SRCS float16_bfloat16_host_test.cc
  DEPS gtest glog)
if(WITH_CUDA)
  nv_test(
    test_fp16_bf16_cuda
    SRCS float16_bfloat16_cuda_test.cu
    DEPS gtest glog)
endif()
