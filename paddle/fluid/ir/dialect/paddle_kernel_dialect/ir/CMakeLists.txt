add_subdirectory(legacy_kernel_dialect)

file(GLOB PADDLE_KERNEL_DIALECT_SRCS "*.cc")
cc_library(
  pd_kernel_dialect
  SRCS ${PADDLE_KERNEL_DIALECT_SRCS}
  DEPS pd_dialect_core)
