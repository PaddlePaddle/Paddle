cc_library(subgraph_detector SRCS subgraph_detector.cc DEPS proto_desc)
if(WITH_TESTING)
  add_dependencies(subgraph_detector gtest)
endif()

if (WITH_GPU AND TENSORRT_FOUND)
  cc_library(tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass.cc DEPS subgraph_detector tensorrt_op_teller)

  set(analysis_deps ${analysis_deps}
          subgraph_detector tensorrt_subgraph_pass
          CACHE INTERNAL "")

  set(pass_file ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h)
  file(APPEND ${pass_file} "USE_PASS(tensorrt_subgraph_pass);\n")
  set(INFER_IR_PASSES ${INFER_IR_PASSES} tensorrt_subgraph_pass CACHE INTERNAL "")
endif()
