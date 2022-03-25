cc_library(cinn_cache_key SRCS cinn_cache_key.cc DEPS boost graph graph_helper lod_tensor proto_desc)
cc_library(build_cinn_pass SRCS build_cinn_pass.cc DEPS pass subgraph_detector graph_pattern_detector cinn_compiler errors enforce)
cc_library(transform_desc SRCS transform_desc.cc DEPS proto_desc cinn)
cc_library(cinn_graph_symbolization SRCS cinn_graph_symbolization.cc DEPS lod_tensor graph transform_desc cinn)
cc_library(cinn_compiler SRCS cinn_compiler.cc DEPS framework_proto graph lod_tensor cinn_cache_key cinn_graph_symbolization cinn cinn_launch_context)

if (WITH_TESTING)
  cc_test(cinn_lib_test SRCS cinn_lib_test.cc DEPS cinn)
  set_tests_properties(cinn_lib_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(cinn_cache_key_test SRCS cinn_cache_key_test.cc DEPS cinn_cache_key)
  set_tests_properties(cinn_cache_key_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(build_cinn_pass_test SRCS build_cinn_pass_test.cc DEPS build_cinn_pass cinn_compiler op_registry mul_op activation_op elementwise_add_op)
  set_tests_properties(build_cinn_pass_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(transform_desc_test SRCS transform_desc_test.cc DEPS transform_desc)
  set_tests_properties(transform_desc_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(cinn_graph_symbolization_test SRCS cinn_graph_symbolization_test.cc DEPS cinn_graph_symbolization)
  set_tests_properties(cinn_graph_symbolization_test PROPERTIES LABELS "RUN_TYPE=CINN")

  cc_test(cinn_compiler_test SRCS cinn_compiler_test.cc DEPS cinn_compiler place proto_desc graph_viz_pass build_cinn_pass cinn mul_op activation_op elementwise_add_op)
  set_tests_properties(cinn_compiler_test PROPERTIES LABELS "RUN_TYPE=CINN")
endif()
