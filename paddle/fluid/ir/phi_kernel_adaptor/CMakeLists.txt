# All source files of pd_dialect, except for the source file of op, which is generated in the compilation directory.
file(GLOB PHI_KERNEL_ADAPTOR_SRCS "*.cc")

cc_library(
  phi_kernel_adaptor
  SRCS ${PHI_KERNEL_ADAPTOR_SRCS}
  DEPS program_translator ir phi_utils)
