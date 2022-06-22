cc_library(
  jit_serializer
  SRCS serializer.cc
  DEPS lod_tensor device_context)

cc_library(
  jit_layer_utils
  SRCS layer_utils.cc
  DEPS scope proto_desc)

cc_library(
  jit_layer
  SRCS layer.cc
  DEPS executor parallel_executor executor_cache)

cc_library(jit_function_schema SRCS function_schema.cc)

if(WITH_TESTING AND NOT WIN32)
  add_custom_target(
    jit_download_program
    COMMAND wget -nc
            https://paddle-ci.gz.bcebos.com/dy2st/export.forward.pdiparams
    COMMAND wget -nc
            https://paddle-ci.gz.bcebos.com/dy2st/export.forward.pdmodel
    COMMAND wget -nc https://paddle-ci.gz.bcebos.com/dy2st/export.infer.pdmodel)
  set(JIT_DEPS
      phi
      elementwise_add_op
      matmul_v2_op
      activation_op
      reduce_mean_op
      feed_op
      fetch_op
      scale_op
      jit_serializer
      jit_layer
      jit_layer_utils
      jit_function_schema)
  cc_test(
    layer_test
    SRCS layer_test.cc
    DEPS ${JIT_DEPS})
  add_dependencies(layer_test jit_download_program)
endif()
