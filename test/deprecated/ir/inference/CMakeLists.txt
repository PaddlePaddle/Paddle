file(
  GLOB TEST_INFERENCE_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_INFERENCE_IR_PASSES "${TEST_INFERENCE_IR_PASSES}")

file(
  GLOB TEST_TRT_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_trt_*.py")
string(REPLACE ".py" "" TEST_TRT_IR_PASSES "${TEST_TRT_IR_PASSES}")

file(
  GLOB TEST_TRT_CONVERTER
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_trt_convert_*.py")
string(REPLACE ".py" "" TEST_TRT_CONVERTER "${TEST_TRT_CONVERTER}")

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

foreach(TEST_INFERENCE_IR_PASS ${TEST_TRT_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

if(WITH_GPU AND TENSORRT_FOUND)

  foreach(TRT_CONVERT ${TEST_TRT_CONVERTER})
    list(REMOVE_ITEM TEST_TRT_IR_PASSES ${TRT_CONVERT})
  endforeach()

  foreach(target ${TEST_TRT_CONVERTER})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES TIMEOUT 300)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

if(WITH_GPU AND TENSORRT_FOUND)
  set_tests_properties(test_trt_subgraph_pass PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_conv_pass PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_dynamic_shape PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_inspector PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_inference_predictor PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_optimization_level PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_ops_fp32_mix_precision PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_inference_fp16_io PROPERTIES TIMEOUT 500)
  set_tests_properties(test_trt_convert_unary PROPERTIES TIMEOUT 600)

  if(WITH_NV_JETSON)
    set_tests_properties(
      test_trt_pool_op
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 450)
    set_tests_properties(
      test_trt_pool3d_op
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 450)
  else()
    set_tests_properties(
      test_trt_pool_op
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 300)
    set_tests_properties(
      test_trt_pool3d_op
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT 45)
  endif()

  set_tests_properties(test_trt_tile_op PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_fc_fuse_quant_dequant_pass PROPERTIES TIMEOUT
                                                                      100)
  set_tests_properties(test_trt_conv_quant_dequant_pass PROPERTIES TIMEOUT 180)
  set_tests_properties(test_trt_matmul_quant_dequant PROPERTIES TIMEOUT 450)
  set_tests_properties(test_trt_conv3d_transpose_op PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_nearest_interp_v2_op PROPERTIES TIMEOUT 30)
endif()
