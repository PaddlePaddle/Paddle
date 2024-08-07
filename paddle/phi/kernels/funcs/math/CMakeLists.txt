file(
  GLOB func_cc_srcs
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
list(FILTER func_cc_srcs EXCLUDE REGEX ".*_xpu\\.cc$")

if(WITH_GPU OR WITH_ROCM)
  file(
    GLOB func_cu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.cu")
endif()

if(WITH_XPU)
  file(
    GLOB func_xpu_srcs
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*_xpu.cc")
endif()

collect_srcs(kernels_srcs SRCS ${func_cc_srcs} ${func_cu_srcs} ${func_xpu_srcs})
