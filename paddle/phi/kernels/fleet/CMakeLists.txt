file(
  GLOB fleet_ops_cu
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cu" "gpu/*.cu" "gpu/*.cu.cc")

file(
  GLOB fleet_ops_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc" "cpu/*.cc")

if(WITH_GPU OR WITH_ROCM)
  collect_srcs(fleet_ops_srcs SRCS ${fleet_ops_cu})
  kernel_declare("${fleet_ops_cu}")
endif()

collect_srcs(fleet_ops_srcs SRCS ${fleet_ops_cc})
kernel_declare("${fleet_ops_cc}")
