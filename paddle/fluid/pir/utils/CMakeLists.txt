cc_library(
  pir_general_functions
  SRCS general_functions.cc analysis_info.cc
  DEPS op_dialect op_dialect_vjp pir)

cc_library(
  pir_name_analysis
  SRCS name_analysis.cc
  DEPS op_dialect op_dialect_vjp pir)
