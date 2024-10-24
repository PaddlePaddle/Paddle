file(GLOB_RECURSE transforms_srcs "*.cc")
if(NOT WITH_CINN)
  list(REMOVE_ITEM transforms_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/build_cinn_pass.cc
       ${CMAKE_CURRENT_SOURCE_DIR}/sub_graph_extract_pass.cc)
endif()

if(NOT WITH_ONEDNN)
  file(GLOB_RECURSE onednn_srcs "onednn/*.cc")
  list(REMOVE_ITEM transforms_srcs ${onednn_srcs})
endif()

if(NOT WITH_XPU)
  file(GLOB_RECURSE xpu_srcs "xpu/*.cc")
  list(REMOVE_ITEM transforms_srcs ${xpu_srcs})
endif()

if(NOT TENSORRT_FOUND)
  file(GLOB_RECURSE trt_srcs "tensorrt/*.cc")
  list(REMOVE_ITEM transforms_srcs ${trt_srcs})
endif()

set(transforms_deps
    drr
    op_dialect
    op_dialect_vjp
    standalone_executor
    pir
    pir_general_functions
    phi
    common)

if(WITH_CINN)
  set(transforms_deps ${transforms_deps} cinnapi)
endif()

cc_library(
  pir_transforms
  SRCS ${transforms_srcs}
  DEPS ${transforms_deps})
