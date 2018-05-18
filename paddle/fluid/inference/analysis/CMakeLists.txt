cc_library(analysis SRCS dot.cc node.cc data_flow_graph.cc graph_traits.cc subgraph_splitter.cc
        data_flow_graph_to_fluid_pass.cc fluid_to_data_flow_graph_pass.cc)
cc_test(test_node SRCS node_tester.cc DEPS analysis)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)
cc_test(test_data_flow_graph SRCS data_flow_graph_tester.cc DEPS analysis ${FLUID_CORE_MODULES})
cc_test(test_subgraph_spliter_tester
        SRCS subgraph_splitter_tester.cc
        DEPS analysis ${FLUID_CORE_MODULES}
        paddle_fluid_api)
