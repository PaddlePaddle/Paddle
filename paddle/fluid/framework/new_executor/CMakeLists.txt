cc_library(interpretercore SRCS interpretercore.cc)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore)

# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler)
