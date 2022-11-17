pass_library(
  build_cinn_pass
  base
  DEPS
  cinn_subgraph_detector
  subgraph_detector
  cinn_compiler
  errors
  enforce)

cc_library(
  cinn_cache_key
  SRCS cinn_cache_key.cc
  DEPS graph graph_helper lod_tensor proto_desc)
cc_library(
  cinn_subgraph_detector
  SRCS cinn_subgraph_detector.cc
  DEPS graph graph_helper subgraph_detector lod_tensor proto_desc)
cc_library(
  transform_desc
  SRCS transform_desc.cc
  DEPS proto_desc cinn)
cc_library(
  transform_type
  SRCS transform_type.cc
  DEPS errors enforce cinn)
cc_library(
  cinn_graph_symbolization
  SRCS cinn_graph_symbolization.cc
  DEPS lod_tensor graph transform_desc cinn)
cc_library(
  cinn_compiler
  SRCS cinn_compiler.cc
  DEPS framework_proto
       graph
       lod_tensor
       cinn_cache_key
       cinn_graph_symbolization
       cinn
       cinn_launch_context)

if(WITH_TESTING)
  cc_test_old(cinn_lib_test SRCS cinn_lib_test.cc DEPS cinn)
  set_tests_properties(cinn_lib_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS
              cinn_cache_key)
  set_tests_properties(cinn_cache_key_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(
    build_cinn_pass_test
    SRCS
    build_cinn_pass_test.cc
    DEPS
    build_cinn_pass
    cinn_compiler
    op_registry
    mul_op
    activation_op
    elementwise_add_op
    generated_op)
  set_tests_properties(build_cinn_pass_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(transform_desc_test SRCS transform_desc_test.cc DEPS
              transform_desc)
  set_tests_properties(transform_desc_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(transform_type_test SRCS transform_type_test.cc DEPS
              transform_type)
  set_tests_properties(transform_type_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test_old(cinn_graph_symbolization_test SRCS
              cinn_graph_symbolization_test.cc DEPS cinn_graph_symbolization)
  set_tests_properties(cinn_graph_symbolization_test PROPERTIES LABELS
                                                                "RUN_TYPE=CINN")

  cc_test_old(
    cinn_compiler_test
    SRCS
    cinn_compiler_test.cc
    DEPS
    cinn_compiler
    place
    proto_desc
    graph_viz_pass
    build_cinn_pass
    cinn
    mul_op
    activation_op
    elementwise_add_op
    generated_op)
  set_tests_properties(cinn_compiler_test PROPERTIES LABELS "RUN_TYPE=CINN")
endif()
