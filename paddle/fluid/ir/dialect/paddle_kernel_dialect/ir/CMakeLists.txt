file(GLOB PADDLE_KERNEL_DIALECT_SRCS "*.cc")
cc_library(
  pd_kernel_dialect NOT_FOR_INFER
  SRCS ${PADDLE_KERNEL_DIALECT_SRCS}
  DEPS pd_dialect_core)
