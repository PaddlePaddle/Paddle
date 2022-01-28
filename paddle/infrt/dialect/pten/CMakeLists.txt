mlir_tablegen_on(infrt_pten_base DIALECT pten)
mlir_tablegen_on(infrt_pten_tensor DIALECT pten_dt)

cc_library(infrt_pten SRCS pten_base.cc DEPS infrt)
