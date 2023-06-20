# All source files of pd_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PD_PASS_SRCS "*.cc")

cc_library(
  pd_op_to_kernel_pass
  SRCS ${PD_PASS_SRCS}
  DEPS ir phi_utils)
