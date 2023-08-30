if(NOT CINN_ONLY)
  cinn_cc_library(runtime_dialect SRCS runtime_dialect.cc jit_kernel_op.cc DEPS
                  op_dialect)
endif()
