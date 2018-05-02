cc_library(
    inference_analysis SRCS node.cc graph_traits.cc helper.h data_flow_graph.cc subgraph_splitter.cc pass.cc tensorrt_subgraph_pass.cc)
