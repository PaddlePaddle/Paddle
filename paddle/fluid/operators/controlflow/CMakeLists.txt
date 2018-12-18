include(operators)
register_operators(DEPS naive_executor)

file(APPEND ${pybind_file} "USE_OP(less_than);\nUSE_OP(logical_and);\nUSE_NO_KERNEL_OP(read_from_array);\n")
