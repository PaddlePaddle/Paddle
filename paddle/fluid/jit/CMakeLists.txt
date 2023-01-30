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
  jit_function
  SRCS function.cc
<<<<<<< HEAD
  DEPS jit_function_utils jit_interpreter_engine jit_predictor_engine)
=======
  DEPS jit_function_utils jit_executor_engine jit_pe_engine
       jit_predictor_engine)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

cc_library(
  jit_layer
  SRCS layer.cc
  DEPS jit_serializer
       jit_function_utils
       jit_serializer_utils
       jit_compilation_unit
       jit_function_schema
<<<<<<< HEAD
       jit_interpreter_engine
=======
       jit_executor_engine
       jit_pe_engine
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
       jit_predictor_engine
       jit_function)

if(WITH_TESTING AND NOT WIN32)
  add_custom_target(
    jit_download_program
    COMMAND
      wget -nc -q --no-check-certificate
      https://paddle-ci.gz.bcebos.com/dy2st/multi_program_load_with_property.tar.gz
<<<<<<< HEAD
    COMMAND tar zxf multi_program_load_with_property.tar.gz
    WORKING_DIRECTORY "${CC_TESTS_DIR}")
=======
    COMMAND tar zxf multi_program_load_with_property.tar.gz)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
=======
  add_dependencies(layer_test jit_download_program)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()
