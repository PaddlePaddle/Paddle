cc_library(
  jit_executor_function
  SRCS executor_function.cc
  DEPS executor)

cc_library(
  jit_pe_function
  SRCS pe_function.cc
  DEPS parallel_executor)
