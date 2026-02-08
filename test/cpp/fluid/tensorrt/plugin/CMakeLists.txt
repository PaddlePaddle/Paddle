if(WIN32)
  nv_test(
    test_split_plugin
    SRCS test_split_plugin.cc
    DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_plugin
         dynload_tensorrt)
else()
  nv_test(
    test_split_plugin
    SRCS test_split_plugin.cc
    DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_plugin)
endif()

if(NOT WIN32)
  nv_test(
    test_fused_token_prune_plugin
    SRCS test_fused_token_prune_plugin.cc
    DEPS paddle_framework ${GLOB_OPERATOR_DEPS} tensorrt_plugin)
endif()
