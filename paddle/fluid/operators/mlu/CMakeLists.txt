
IF(WITH_MLU)
    cc_library(mlu_baseop SRCS mlu_baseop.cc DEPS neuware_lib framework_proto xxhash)
    cc_test(activation_op_mlu_test SRCS activation_op_mlu_test.cc DEPS op_registry activation_op scope device_context executor)
ENDIF()
