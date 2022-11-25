cc_test(
  op_tester
<<<<<<< HEAD
  SRCS op_tester.cc
=======
  SRCS op_tester.cc op_tester_config.cc
>>>>>>> 5b0760feb220cd8f9e8a247c638a0f0d6df64baf
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
