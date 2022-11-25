<<<<<<< HEAD
add_subdirectory(engine)
proto_library(paddle_jit_property_proto SRCS property.proto)

cc_library(
  jit_property
  SRCS property.cc
  DEPS paddle_jit_property_proto tensor)

cc_library(
  jit_serializer
  SRCS serializer.cc
  DEPS lod_tensor device_context jit_property)
=======
cc_library(
  jit_serializer
  SRCS serializer.cc
  DEPS lod_tensor device_context)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

cc_library(
  jit_function_utils
  SRCS function_utils.cc
  DEPS lod_tensor scope proto_desc)

cc_library(
  jit_serializer_utils
  SRCS serializer_utils.cc
  DEPS proto_desc)

cc_library(
  jit_compilation_unit
  SRCS compilation_unit.cc
  DEPS proto_desc executor parallel_executor executor_cache)

cc_library(
  jit_function_schema
  SRCS function_schema.cc
  DEPS jit_function_utils)

cc_library(
<<<<<<< HEAD
  jit_function
  SRCS function.cc
  DEPS jit_function_utils jit_interpreter_engine jit_predictor_engine)

cc_library(
  jit_layer
  SRCS layer.cc
  DEPS jit_serializer
       jit_function_utils
       jit_serializer_utils
       jit_compilation_unit
       jit_function_schema
       jit_interpreter_engine
       jit_predictor_engine
       jit_function)
=======
  jit_layer
  SRCS layer.cc
  DEPS jit_serializer jit_function_utils jit_serializer_utils
       jit_compilation_unit jit_function_schema)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

if(WITH_TESTING AND NOT WIN32)
  add_custom_target(
    jit_download_program
<<<<<<< HEAD
    COMMAND
      wget -nc -q --no-check-certificate
      https://paddle-ci.gz.bcebos.com/dy2st/multi_program_load_with_property.tar.gz
    COMMAND tar zxf multi_program_load_with_property.tar.gz
    WORKING_DIRECTORY "${CC_TESTS_DIR}")
=======
    COMMAND wget -nc -q --no-check-certificate
            https://paddle-ci.gz.bcebos.com/dy2st/multi_program_load.tar.gz
    COMMAND tar zxf multi_program_load.tar.gz)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
  set(JIT_DEPS
      phi
      phi_api
      elementwise_add_op
      matmul_v2_op
      activation_op
      reduce_mean_op
      feed_op
      fetch_op
      scale_op
<<<<<<< HEAD
      transfer_layout_op
=======
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
      jit_layer)
  cc_test(
    layer_test
    SRCS layer_test.cc
    DEPS ${JIT_DEPS})
<<<<<<< HEAD
  # add_dependencies(layer_test jit_download_program)

  cc_test(
    layer_test_new
    SRCS layer_test.cc
    DEPS ${JIT_DEPS})
  # add_dependencies(layer_test_new jit_download_program)
  # set_tests_properties(layer_test_new PROPERTIES ENVIRONMENT
  #                                                "FLAGS_jit_engine_type=New")
endif()
=======
  add_dependencies(layer_test jit_download_program)
endif()

proto_library(paddle_jit_property_proto SRCS property.proto)

cc_library(
  jit_property
  SRCS property.cc
  DEPS paddle_jit_property_proto)
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
