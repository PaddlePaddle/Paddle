cc_test(op_tester SRCS op_tester.cc op_tester_config.cc
        DEPS memory timer framework_proto proto_desc lod_tensor op_registry
        device_context scope glob_op eigen_function)
