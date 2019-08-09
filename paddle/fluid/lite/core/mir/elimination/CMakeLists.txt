if (NOT LITE_WITH_LIGHT_WEIGHT_FRAMEWORK)
  lite_cc_test(test_identity_scale_eliminate_pass_lite
    SRCS identity_scale_eliminate_pass_test.cc
    DEPS mir_passes program_lite proto_desc cpp_op_desc_lite
    ${ops_lite}
    )
endif()
