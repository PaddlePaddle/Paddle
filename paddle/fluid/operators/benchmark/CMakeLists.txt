cc_test(
  op_tester
<<<<<<< HEAD
  SRCS op_tester.cc
=======
  SRCS op_tester.cc op_tester_config.cc
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e
  DEPS memory
       timer
       framework_proto
       proto_desc
       lod_tensor
       op_registry
       device_context
       scope
       ${GLOB_OP_LIB}
       ${GLOB_OPERATOR_DEPS}
       eigen_function)

if(WITH_ONNXRUNTIME AND WIN32)
  # Copy onnxruntime for some c++ test in Windows, since the test will
  # be build only in CI, so suppose the generator in Windows is Ninja.
  copy_onnx(op_tester)
endif()
