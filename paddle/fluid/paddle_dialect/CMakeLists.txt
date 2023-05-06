set(PD_DIALECT_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/fluid/paddle_dialect")
set(PD_DIALECT_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/fluid/paddle_dialect")

file(GLOB PD_DIALECT_SRCS "*.cc")

cc_library(
  paddle_dialect
  SRCS ${PD_DIALECT_SRCS}
  DEPS new_ir)
