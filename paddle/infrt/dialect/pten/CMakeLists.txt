#mlir_tablegen_on(infrt_pten_base DIALECT pten)
add_mlir_dialect(infrt_pten_base pten)
add_mlir_dialect(infrt_pten_tensor pten_dt)
#mlir_tablegen_on(infrt_pten_tensor)

cc_library(infrt_pten SRCS pten_base.cc DEPS infrt)
