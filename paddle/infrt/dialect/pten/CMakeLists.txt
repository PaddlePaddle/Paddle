if (NOT INFRT_WITH_PTEN)
    return()
endif()

#mlir_tablegen_on(infrt_pten_base DIALECT pten)
add_mlir_dialect(infrt_pten_base pten)
add_mlir_dialect(infrt_pten_tensor pten_dt)
#mlir_tablegen_on(infrt_pten_tensor)

gather_srcs(infrt_src SRCS
    pten_base.cc infrt_pten_tensor.cc
    infrt_pten_tensor.cc)
