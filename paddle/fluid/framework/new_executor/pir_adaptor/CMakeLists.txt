# All source files of pd_op_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PIR_ADAPTOR_SRCS "*.cc")

cc_library(
  pir_adaptor
  SRCS ${PIR_ADAPTOR_SRCS}
  DEPS program_translator pd_kernel_dialect)
