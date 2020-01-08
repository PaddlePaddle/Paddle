cc_library(subgraph_util SRCS subgraph_util.cc DEPS subgraph_detector)

if (WITH_GPU AND TENSORRT_FOUND)
  cc_library(tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass.cc DEPS subgraph_util tensorrt_op_teller)

  set(analysis_deps ${analysis_deps}
          subgraph_util tensorrt_subgraph_pass
          CACHE INTERNAL "")

  set(pass_file ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h)
  file(APPEND ${pass_file} "USE_PASS(tensorrt_subgraph_pass);\n")
  set(INFER_IR_PASSES ${INFER_IR_PASSES} tensorrt_subgraph_pass CACHE INTERNAL "")
endif()

if (ANAKIN_SUBGRAPH) 
  cc_library(anakin_subgraph_pass SRCS anakin_subgraph_pass.cc DEPS subgraph_util anakin_op_teller)

  set(analysis_deps ${analysis_deps}
          subgraph_util anakin_subgraph_pass
          CACHE INTERNAL "")

  set(pass_file ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h)
  file(APPEND ${pass_file} "USE_PASS(anakin_subgraph_pass);\n")
  set(INFER_IR_PASSES ${INFER_IR_PASSES} anakin_subgraph_pass CACHE INTERNAL "")
endif()
