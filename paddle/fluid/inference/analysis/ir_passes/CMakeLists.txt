cc_library(subgraph_detector SRCS subgraph_detector.cc DEPS proto_desc)
cc_library(tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass.cc DEPS subgraph_detector)
cc_library(mem_optim_pass SRCS mem_optim_pass.cc DEPS graph)

set(analysis_deps ${analysis_deps}
        subgraph_detector
        tensorrt_subgraph_pass
        mem_optim_pass
        CACHE INTERNAL "")

set(pass_file ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h)
file(APPEND ${pass_file} "USE_PASS(tensorrt_subgraph_pass);\n")
file(APPEND ${pass_file} "USE_PASS(mem_optim_pass);\n")

set(INFER_IR_PASSES ${INFER_IR_PASSES}
        tensorrt_subgraph_pass
        mem_optim_pass
        CACHE INTERNAL "")
