file(
  GLOB TEST_INFERENCE_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INFERENCE_IR_PASSES "${TEST_INFERENCE_IR_PASSES}")

# Only for cpu(mkl + openblas)
set(TEST_INFERENCE_CPU_UT "test_mul_lstm_fuse_pass" "test_mul_gru_fuse_pass")

if(NOT WITH_MKLDNN
   AND NOT TENSORRT_FOUND
   AND NOT WITH_GPU)
  foreach(target ${TEST_INFERENCE_CPU_UT})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()

  set_tests_properties(test_mul_lstm_fuse_pass PROPERTIES TIMEOUT 1000)
  set_tests_properties(test_mul_gru_fuse_pass PROPERTIES TIMEOUT 600)
endif()
