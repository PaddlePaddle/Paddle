set(TENSORRT_VERSION_NUMBER
    "${TENSORRT_MAJOR_VERSION}${TENSORRT_MINOR_VERSION}")
if(${TENSORRT_VERSION_NUMBER} GREATER_EQUAL 85)
  nv_test(
    test_tensorrt_engine_instruction
    SRCS test_tensorrt_engine_instruction.cc
    DEPS pir
         trt_engine
         naive_executor
         phi
         common
         pir_save_load
         pir_tensorrt_plugin)
  set_tests_properties(test_tensorrt_engine_instruction PROPERTIES TIMEOUT 120)
  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(test_tensorrt_engine_instruction)
  endif()
endif()
