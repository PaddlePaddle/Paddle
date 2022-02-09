core_gather_headers()

gather_srcs(infrt_src SRCS
    dialect.cc
    basic_kernels.cc
    test_kernels.cc
    infrt_base.cc
    init_infrt_dialects.cc
    tensor_shape.cc
    dense_tensor.cc
    mlir_loader.cc
    diagnostic_utils.cc
    pd_types.cc
    pd_ops.cc
    )

mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(test_kernels)
mlir_tablegen_on(infrt_base DIALECT infrt)
mlir_tablegen_on(tensor_shape DIALECT ts)
mlir_tablegen_on(dense_tensor DIALECT dt)
mlir_tablegen_on(pd_op_base DIALECT pd)
mlir_tablegen_on(pd_ops)
mlir_tablegen_on(pd_extra_ops)
mlir_add_rewriter(rewrite)

# TODO(Superjomn) add a cmake function cc_executable to ecapsulate the following code
add_executable(infrtopt opt.cc)
target_link_libraries(infrtopt infrt)

add_executable(print-ir print_ir.cc)
target_link_libraries(print-ir infrt ${mlir_libs})
add_dependencies(print-ir pd_ops_inc)

cc_test_tiny(test_infrt_mlir_loader SRCS mlir_loader_test.cc DEPS infrt ${MLIR_IR_LIBS})

add_subdirectory(tensorrt)

if (INFRT_WITH_PTEN)
    add_subdirectory(pten)
endif()
