if(NOT CINN_ONLY)
  cinn_cc_library(cinn_runtime_dialect SRCS runtime_dialect.cc jit_kernel_op.cc
                  DEPS pir_core)
endif()
