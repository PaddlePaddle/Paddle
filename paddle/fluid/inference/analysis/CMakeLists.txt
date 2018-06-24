set(FLUID_CORE_MODULES proto_desc memory lod_tensor executor init)
cc_library(analysis SRCS pass_manager.cc dot.cc node.cc data_flow_graph.cc graph_traits.cc subgraph_splitter.cc
  fluid_to_data_flow_graph_pass.cc
  data_flow_graph_to_fluid_pass.cc
  tensorrt_subgraph_pass.cc
  dfg_graphviz_draw_pass.cc
  DEPS framework_proto)
cc_test(test_node SRCS node_tester.cc DEPS analysis)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)

set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)

function (inference_analysis_test TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS)
    cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    cc_test(${TARGET}
            SRCS "${analysis_test_SRCS}"
            DEPS analysis
            ARGS --inference_model_dir=${PYTHON_TESTS_DIR}/book/word2vec.inference.model --fraction_of_gpu_memory_to_use=0.5)
    set_tests_properties(${TARGET} PROPERTIES DEPENDS test_word2vec)
endfunction(inference_analysis_test)

inference_analysis_test(test_data_flow_graph SRCS data_flow_graph_tester.cc)
inference_analysis_test(test_data_flow_graph_to_fluid_pass SRCS data_flow_graph_to_fluid_pass_tester.cc)
inference_analysis_test(test_fluid_to_data_flow_graph_pass SRCS fluid_to_data_flow_graph_pass_tester.cc)
inference_analysis_test(test_subgraph_splitter SRCS subgraph_splitter_tester.cc)
inference_analysis_test(test_dfg_graphviz_draw_pass SRCS dfg_graphviz_draw_pass_tester.cc)
#inference_analysis_test(test_tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass_tester.cc)
inference_analysis_test(test_pass_manager SRCS pass_manager_tester.cc)
