# TODO(Aurelius84): new_ir_compiler depends on pd_dialect and could
# not found under CINN_ONLY mode
if(NOT CINN_ONLY)
  cinn_cc_library(cinn_dialect SRCS runtime_dialect.cc jit_kernel_op.cc DEPS
                  pd_dialect)
endif()
