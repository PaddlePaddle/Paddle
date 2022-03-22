core_gather_headers()

gather_srcs(infrt_src SRCS
    dialect.cc
    init_dialects.cc
    tensor_shape.cc
    dense_tensor.cc
    mlir_loader.cc
    diagnostic_utils.cc
    )

mlir_tablegen_on(tensor_shape DIALECT ts)
mlir_tablegen_on(dense_tensor DIALECT dt)

# TODO(Superjomn) add a cmake function cc_executable to ecapsulate the following code
add_executable(infrtopt opt.cc)
target_link_libraries(infrtopt infrt)

add_executable(print-ir print_ir.cc)
target_link_libraries(print-ir infrt ${mlir_libs})
cc_test_tiny(test_infrt_mlir_loader SRCS mlir_loader_test.cc DEPS infrt ${MLIR_IR_LIBS})

add_subdirectory(infrt)
add_subdirectory(pd)
add_subdirectory(tensorrt)

if (INFRT_WITH_PHI)
    add_subdirectory(phi)
endif()
