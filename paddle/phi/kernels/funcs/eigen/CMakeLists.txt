file(
  GLOB EIGEN_CC_SOURCES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc")
file(
  GLOB EIGEN_CU_SOURCES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cu")

collect_srcs(kernels_srcs SRCS ${EIGEN_CC_SOURCES} ${EIGEN_CU_SOURCES})
