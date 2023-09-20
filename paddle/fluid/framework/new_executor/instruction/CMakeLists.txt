cc_library(
  instruction_base
  SRCS instruction_base.cc phi_kernel_instruction.cc
       legacy_kernel_instruction.cc cond_instruction.cc instruction_util.cc
  DEPS phi framework_proto)

if(WITH_CINN AND NOT CINN_ONLY)
  cc_library(
    cinn_jit_instruction NOT_FOR_INFER
    SRCS cinn_jit_instruction.cc
    DEPS phi cinnapi cinn_op_dialect cinn_runtime_dialect)
endif()
