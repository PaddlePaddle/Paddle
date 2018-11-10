cc_library(ir_pass_manager SRCS ir_pass_manager.cc DEPS graph pass)
set(analysis_deps
        framework_proto proto_desc ir_pass_manager graph pass paddle_fluid_api executor pretty_log)

cc_library(analysis SRCS pass_manager.cc node.cc data_flow_graph.cc graph_traits.cc subgraph_splitter.cc
  analyzer.cc
  helper.cc
  # passes
  analysis_pass.cc
  fluid_to_data_flow_graph_pass.cc
  data_flow_graph_to_fluid_pass.cc
  dfg_graphviz_draw_pass.cc
  tensorrt_subgraph_pass.cc
  tensorrt_subgraph_node_mark_pass.cc
  fluid_to_ir_pass.cc
  model_store_pass.cc
  DEPS ${analysis_deps})

cc_test(test_node SRCS node_tester.cc DEPS analysis)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)
cc_binary(inference_analyzer SRCS analyzer_main.cc DEPS analysis paddle_fluid)

function(inference_analysis_test TARGET)
  if(WITH_TESTING)
     set(options "")
     set(oneValueArgs "")
     set(multiValueArgs SRCS ARGS EXTRA_DEPS)
     cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
     inference_base_test(${TARGET}
             SRCS ${analysis_test_SRCS}
             DEPS analysis pass ${GLOB_PASS_LIB} ${analysis_test_EXTRA_DEPS}
             ARGS --inference_model_dir=${WORD2VEC_MODEL_DIR} ${analysis_test_ARGS})
  endif()
endfunction(inference_analysis_test)

inference_analysis_test(test_analyzer SRCS analyzer_tester.cc EXTRA_DEPS paddle_inference_api)
inference_analysis_test(test_data_flow_graph SRCS data_flow_graph_tester.cc)
inference_analysis_test(test_data_flow_graph_to_fluid_pass SRCS data_flow_graph_to_fluid_pass_tester.cc)
inference_analysis_test(test_fluid_to_ir_pass SRCS fluid_to_ir_pass_tester.cc)
inference_analysis_test(test_fluid_to_data_flow_graph_pass SRCS fluid_to_data_flow_graph_pass_tester.cc)
inference_analysis_test(test_subgraph_splitter SRCS subgraph_splitter_tester.cc)
inference_analysis_test(test_dfg_graphviz_draw_pass SRCS dfg_graphviz_draw_pass_tester.cc)
inference_analysis_test(test_tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass_tester.cc)
inference_analysis_test(test_pass_manager SRCS pass_manager_tester.cc)
inference_analysis_test(test_tensorrt_subgraph_node_mark_pass SRCS tensorrt_subgraph_node_mark_pass_tester.cc)
inference_analysis_test(test_model_store_pass SRCS model_store_pass_tester.cc)
