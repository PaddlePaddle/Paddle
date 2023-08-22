file(GLOB LEGACY_KERNEL_DIALECT_SRCS "*.cc")
cc_library(
  legacy_kernel_dialect
  SRCS ${LEGACY_KERNEL_DIALECT_SRCS}
  DEPS pd_dialect_core)
