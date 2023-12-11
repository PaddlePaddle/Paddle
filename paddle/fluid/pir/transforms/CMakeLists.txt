file(GLOB_RECURSE transforms_srcs "*.cc")
if(NOT WITH_CINN)
  list(
    REMOVE_ITEM
    transforms_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/build_cinn_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_extract_pass.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_detector.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shape_optimization_pass.cc)
endif()

set(transforms_deps drr op_dialect op_dialect_vjp standalone_executor pir)

if(WITH_CINN)
  set(transforms_deps ${transforms_deps} cinn_op_dialect cinnapi)
endif()

cc_library(
  pir_transforms
  SRCS ${transforms_srcs}
  DEPS ${transforms_deps})
