cc_library(subgraph_detector SRCS subgraph_detector.cc DEPS proto_desc ir_pass_manager)
cc_library(tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass.cc DEPS subgraph_detector ir_pass_manager)
set(analysis_deps ${analysis_deps}
        subgraph_detector tensorrt_subgraph_pass
        CACHE INTERNAL "")
