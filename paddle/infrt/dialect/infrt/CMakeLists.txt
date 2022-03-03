core_gather_headers()

gather_srcs(infrt_src SRCS
    common_type.cc
    infrt_dialect.cc
    )


add_mlir_dialect(infrt_ops infrt)

set(LLVM_TARGET_DEFINITIONS infrt_ops.td)
mlir_tablegen(infrt_opsAttributes.h.inc -gen-attrdef-decls -dialect=infrt)
mlir_tablegen(infrt_opsAttributes.cpp.inc -gen-attrdef-defs -dialect=infrt)
add_public_tablegen_target(MLIRinfrt_opsAttributesIncGen)
add_dependencies(mlir-headers MLIRinfrt_opsAttributesIncGen)
