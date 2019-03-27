include(operators)
register_operators(DEPS naive_executor)
cc_library(while_op_helper SRCS while_op_helper.cc DEPS operator) 

file(APPEND ${pybind_file} "USE_OP(less_than);\nUSE_OP(logical_and);\nUSE_NO_KERNEL_OP(read_from_array);\n")
