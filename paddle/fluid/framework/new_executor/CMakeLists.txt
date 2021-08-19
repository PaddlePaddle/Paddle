include(operators)

# op_library(fetch2_op SRCS fetch2_op.cc)
cc_library(interpretercore SRCS interpretercore.cc DEPS operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} ${PYBIND_DEPS} profiler fetch2_op)
cc_library(standalone_executor SRCS standalone_executor.cc DEPS interpretercore operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} ${PYBIND_DEPS} profiler fetch2_op)

# cc_binary(standalone_executor_test SRCS standalone_executor_test.cc DEPS interpretercore standalone_executor operator op_registry executor ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS} profiler fetch2_op)
