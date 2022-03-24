include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/controlflow.
    include(unity_build_rule.cmake)
endif()
register_operators(EXCLUDES conditional_block_op DEPS naive_executor)

cc_library(conditional_block_op SRCS conditional_block_op.cc DEPS executor)
cc_library(op_variant SRCS op_variant.cc DEPS operator proto_desc)
cc_library(conditional_block_op_helper SRCS conditional_block_op_helper.cc DEPS operator op_variant conditional_block_op)
cc_library(recurrent_op_helper SRCS recurrent_op_helper.cc DEPS operator op_variant recurrent_op)
cc_library(while_op_helper SRCS while_op_helper.cc DEPS operator op_variant) 

cc_test(conditional_block_op_test SRCS conditional_block_op_test.cc DEPS conditional_block_op executor)

if(WITH_UNITY_BUILD)
    target_link_libraries(paddle_operators_controlflow_unity conditional_block_op)
else()
    target_link_libraries(conditional_block_infer_op conditional_block_op)
endif()

file(APPEND ${pybind_file} "USE_OP_ITSELF(less_than);\nUSE_OP_ITSELF(equal_all);\nUSE_NO_KERNEL_OP(read_from_array);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF(logical_and);\nUSE_OP_ITSELF(logical_or);\nUSE_OP_ITSELF(logical_xor);\nUSE_OP_ITSELF(logical_not);\n")
file(APPEND ${pybind_file} "USE_OP_ITSELF(bitwise_and);\nUSE_OP_ITSELF(bitwise_or);\nUSE_OP_ITSELF(bitwise_xor);\nUSE_OP_ITSELF(bitwise_not);\n")
