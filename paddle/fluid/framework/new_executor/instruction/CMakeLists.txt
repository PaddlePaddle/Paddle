cc_library(
  instruction_base
  SRCS instruction_base.cc phi_kernel_instruction.cc
       legacy_kernel_instruction.cc instruction_util.cc
  DEPS phi framework_proto)
