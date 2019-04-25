cc_library(op_test SRCS op_test.cc op_tester_config.cc
        DEPS memory timer framework_proto proto_desc lod_tensor op_registry
        device_context scope ${GLOB_OP_LIB} ${GLOB_OPERATOR_DEPS})
cc_test(op_tester SRCS op_tester.cc DEPS op_test)
