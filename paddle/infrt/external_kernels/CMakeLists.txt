set(external_kernels_src "basic_kernels.cc")

cc_library(external_kernels SHARED SRCS ${external_kernels_src})
set_target_properties(external_kernels PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

set(basic_mlir "${CMAKE_CURRENT_SOURCE_DIR}/basic.mlir")
set(external_kernels_lib "${CMAKE_CURRENT_BINARY_DIR}/libexternal_kernels.so")
message(STATUS "basic_mlir: ${basic_mlir}")
message(STATUS "external_kernels_lib: ${external_kernels_lib}")
add_test(
    NAME run_and_check_external_kernels
    COMMAND sh -c "${CMAKE_BINARY_DIR}/infrt/host_context/infrt-exec -i ${basic_mlir} --shared_libs=${external_kernels_lib} | ${LLVM_PATH}/bin/FileCheck ${basic_mlir}"
)
