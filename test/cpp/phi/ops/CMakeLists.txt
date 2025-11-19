if(WIN32)
  cc_test(
    test_op_signature
    SRCS test_op_signature.cc
    DEPS type_info common)
else()
  cc_test(
    test_op_signature
    SRCS test_op_signature.cc
    DEPS phi common)
endif()
