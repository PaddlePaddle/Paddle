include(operators)
register_operators()

file(APPEND ${pybind_file} "USE_OP(less_than);\nUSE_OP(logical_and);\nUSE_NO_KERNEL_OP(read_from_array);\n")
