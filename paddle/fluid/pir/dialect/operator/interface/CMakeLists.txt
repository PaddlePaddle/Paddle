# All source files of pd_op_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PD_INTERFACE_SRCS "*.cc")

cc_library(
  pd_interface
  SRCS ${PD_INTERFACE_SRCS}
  DEPS pir_core phi_utils)
