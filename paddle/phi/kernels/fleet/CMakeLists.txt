file(
  GLOB fleet_ops_cu
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cu" "gpu/*.cu" "gpu/*.cu.cc")

file(
  GLOB fleet_ops_cc
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "*.cc" "cpu/*.cc")

set(DISTRIBUTE_COMPILE_FLAGS
    "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor -Wno-error=parentheses"
)
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
  set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
endif()

set(DISTRIBUTE_OPS
    cpu/distributed_lookup_table_kernel.cc
    gpu/distributed_lookup_table_kernel.cu
    cpu/distributed_push_sparse_kernel.cc gpu/distributed_push_sparse_kernel.cu)
foreach(src ${DISTRIBUTE_OPS})
  set_source_files_properties(${src} PROPERTIES COMPILE_FLAGS
                                                ${DISTRIBUTE_COMPILE_FLAGS})
endforeach()

# When WITH_PSLIB is defined, distributed operators are not used.
if((NOT WITH_PSCORE) OR WITH_PSLIB)
  list(REMOVE_ITEM fleet_ops_cc ${DISTRIBUTE_OPS})
  list(REMOVE_ITEM fleet_ops_cu ${DISTRIBUTE_OPS})
endif()

if(WITH_GPU OR WITH_ROCM)
  collect_srcs(fleet_ops_srcs SRCS ${fleet_ops_cu})
  kernel_declare("${fleet_ops_cu}")
endif()
collect_srcs(fleet_ops_srcs SRCS ${fleet_ops_cc})
kernel_declare("${fleet_ops_cc}")
