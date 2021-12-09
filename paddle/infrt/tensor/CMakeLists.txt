core_gather_headers()

gather_srcs(infrt_src SRCS
  tensor_map.cc
  tensor_metadata.cc
  dense_tensor_view.cc
  dense_host_tensor.cc
  tensor_shape.cc
  )

# set(tensor_map_mlir "${CMAKE_SOURCE_DIR}/infrt/dialect/mlir_tests/tensor_map.mlir")
# set(external_kernels_lib "${CMAKE_BINARY_DIR}/paddle/libexternal_kernels.so")
# message(STATUS "tensor_map_mlir: ${tensor_map_mlir}")
# message(STATUS "external_kernels_lib: ${external_kernels_lib}")

# Disable temporarily for the external-kernel's mkldnn is outdate
# add_test(
#     NAME run_and_check_tensor_map
#     COMMAND sh -c "sed -e 's|/infrt/build|${CMAKE_BINARY_DIR}|' ${tensor_map_mlir} > /tmp/tensor_map.mlir && ${CMAKE_BINARY_DIR}/infrt/host_context/infrt-exec -i /tmp/tensor_map.mlir --shared_libs=${external_kernels_lib} | ${LLVM_PATH}/bin/FileCheck ${tensor_map_mlir}"
# )
