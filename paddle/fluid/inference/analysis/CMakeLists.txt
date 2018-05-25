set(FLUID_CORE_MODULES proto_desc memory lod_tensor executor init)
cc_library(analysis SRCS dot.cc node.cc data_flow_graph.cc graph_traits.cc subgraph_splitter.cc fluid_to_data_flow_graph_pass.cc
  DEPS paddle_fluid)
cc_test(test_node SRCS node_tester.cc DEPS analysis)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)

set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)

cc_test(test_data_flow_graph SRCS data_flow_graph_tester.cc DEPS analysis ${FLUID_CORE_MODULES} paddle_fluid
  ARGS --inference_model_dir=${PYTHON_TESTS_DIR}/book/word2vec.inference.model)
set_tests_properties(test_data_flow_graph PROPERTIES DEPENDS test_word2vec)

cc_test(test_subgraph_splitter
        SRCS subgraph_splitter_tester.cc
        DEPS analysis paddle_fluid tensor
        ARGS --inference_model_dir=${PYTHON_TESTS_DIR}/book/word2vec.inference.model)
set_tests_properties(test_subgraph_splitter PROPERTIES DEPENDS test_word2vec)
