cc_library(
  subgraph_util
  SRCS subgraph_util.cc
  DEPS subgraph_detector)

if(WITH_GPU AND TENSORRT_FOUND)
  cc_library(
    tensorrt_subgraph_pass
    SRCS tensorrt_subgraph_pass.cc
    DEPS convert_to_mixed_precision subgraph_util tensorrt_op_teller
         infer_io_utils)

  set(analysis_deps
      ${analysis_deps} subgraph_util tensorrt_subgraph_pass
      CACHE INTERNAL "")

  set(pass_file
      ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h.tmp
  )
  file(APPEND ${pass_file} "USE_PASS(tensorrt_subgraph_pass);\n")
  set(INFER_IR_PASSES
      ${INFER_IR_PASSES} tensorrt_subgraph_pass
      CACHE INTERNAL "")
endif()

if(WITH_OPENVINO)
  cc_library(
    openvino_subgraph_pass
    SRCS openvino_subgraph_pass.cc
    DEPS subgraph_util infer_io_utils)

  set(analysis_deps
      ${analysis_deps} subgraph_util openvino_subgraph_pass
      CACHE INTERNAL "")

  set(pass_file
      ${PADDLE_BINARY_DIR}/paddle/fluid/inference/api/paddle_inference_pass.h.tmp
  )
  file(APPEND ${pass_file} "USE_PASS(openvino_subgraph_pass);\n")
  set(INFER_IR_PASSES
      ${INFER_IR_PASSES} openvino_subgraph_pass
      CACHE INTERNAL "")
endif()
