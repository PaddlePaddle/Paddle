file(GLOB PADDLE_KERNEL_DIALECT_SRCS "*.cc")
cc_library(
  pd_kernel_dialect
  SRCS ${PADDLE_KERNEL_DIALECT_SRCS}
  DEPS pd_op_dialect_core)
