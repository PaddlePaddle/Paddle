core_gather_headers()

gather_srcs(infrt_src SRCS
    core_dialect.cc
    basic_kernels.cc
    test_kernels.cc
    )

add_mlir_dialect(core_ops core)

set(LLVM_TARGET_DEFINITIONS core_ops.td)
mlir_tablegen(core_opsAttributes.h.inc -gen-attrdef-decls -dialect=core)
mlir_tablegen(core_opsAttributes.cpp.inc -gen-attrdef-defs -dialect=core)
add_public_tablegen_target(MLIRcore_opsAttributesIncGen)
add_dependencies(mlir-headers MLIRcore_opsAttributesIncGen)

mlir_tablegen_on(basic_kernels)
mlir_tablegen_on(test_kernels)
