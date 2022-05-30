core_gather_headers()

gather_srcs(infrt_src SRCS
    infrt_dialect.cc
    basic_kernels.cc
    test_kernels.cc
    )

add_mlir_dialect(infrt_ops infrt)

set(LLVM_TARGET_DEFINITIONS infrt_ops.td)
mlir_tablegen(infrt_opsAttributes.h.inc -gen-attrdef-decls -dialect=infrt)
mlir_tablegen(infrt_opsAttributes.cpp.inc -gen-attrdef-defs -dialect=infrt)
add_public_tablegen_target(MLIRinfrt_opsAttributesIncGen)
add_dependencies(mlir-headers MLIRinfrt_opsAttributesIncGen)

mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(test_kernels)
