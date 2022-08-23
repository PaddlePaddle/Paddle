cc_library(
  jit_executor_engine
  SRCS executor_engine.cc
  DEPS executor)

cc_library(
  jit_pe_engine
  SRCS pe_engine.cc
  DEPS parallel_executor)

cc_library(
  jit_new_engine
  SRCS new_engine.cc
  DEPS standalone_executor)
