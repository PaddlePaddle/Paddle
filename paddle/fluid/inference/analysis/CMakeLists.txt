cc_library(
    inference_analysis SRCS node.cc graph_traits.cc helper.h data_flow_graph.cc
        subgraph_splitter.cc pass.cc tensorrt_subgraph_pass.cc
        pass_manager.cc
        data_flow_graph_to_fluid_pass.cc
        fluid_to_data_flow_graph_pass.cc
        dot.cc
        dot.h
        ut_helper.h
)

cc_test(analysis_test SRCS
        node_tester.cc
        dot_tester.cc
        data_flow_graph_tester.cc
        subgraph_splitter_tester.cc
        fluid_to_data_flow_graph_pass_tester.cc
        data_flow_graph_to_fluid_pass_tester.cc
        DEPS inference_analysis framework_proto
        proto_desc
        gtest
        executor
        scope
        place
        paddle_fluid_api
        glog
        )
