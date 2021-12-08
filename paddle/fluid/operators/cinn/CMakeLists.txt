include(operators)
register_operators(EXCLUDES cinn_launch_op)

op_library(cinn_launch_op SRCS cinn_launch_op.cc cinn_launch_op.cu.cc DEPS transform_desc cinn_compiler cinn)
cc_test(cinn_launch_op_test SRCS cinn_launch_op_test.cc DEPS cinn_compiler cinn_launch_op elementwise_add_op)
set_tests_properties(cinn_launch_op_test PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
