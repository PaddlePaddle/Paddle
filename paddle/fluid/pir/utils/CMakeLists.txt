cc_library(
  pir_general_functions
  SRCS general_functions.cc
  DEPS op_dialect op_dialect_vjp pir)
