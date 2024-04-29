unset(analysis_deps CACHE)
set(analysis_deps # analysis_deps can be extended across the project
    framework_proto
    proto_desc
    graph
    pass
    paddle_inference_io
    executor
    pretty_log
    ir_pass_manager
    CACHE INTERNAL "")

add_subdirectory(ir_passes)
add_subdirectory(passes)

cc_library(
  analysis_helper
  SRCS helper.cc
  DEPS framework_proto proto_desc graph paddle_inference_io)

cc_library(
  ir_pass_manager
  SRCS ir_pass_manager.cc pass_result_info.cc
  DEPS graph pass ${INFER_IR_PASSES} analysis_helper)

cc_library(
  argument INTERFACE
  SRCS argument.cc
  DEPS scope proto_desc)
cc_library(
  analysis_pass INTERFACE
  SRCS analysis_pass.cc
  DEPS proto_desc)

cc_library(
  analysis
  SRCS analyzer.cc
  DEPS ${analysis_deps} analysis_helper analysis_pass ${INFER_IR_PASSES})
