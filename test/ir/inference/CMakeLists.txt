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

if(WIN32)
  list(REMOVE_ITEM TEST_TRT_IR_PASSES
       "test_trt_explicit_quantization_mobilenet")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_explicit_quantization_mobilenet")
endif()

foreach(TEST_INFERENCE_IR_PASS ${TEST_TRT_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

if(WITH_GPU AND TENSORRT_FOUND)
  list(REMOVE_ITEM TEST_TRT_IR_PASSES test_trt_multiclass_nms_op)

  foreach(TRT_CONVERT ${TEST_TRT_CONVERTER})
    list(REMOVE_ITEM TEST_TRT_IR_PASSES ${TRT_CONVERT})
  endforeach()

  foreach(target ${TEST_TRT_IR_PASSES})
    if(${target} STREQUAL "test_trt_slice_dynamic_plugin")
      if("${TENSORRT_MAJOR_VERSION}.${TENSORRT_MINOR_VERSION}" VERSION_GREATER
         "7.1")
        py_test_modules(${target} MODULES ${target})
        set_tests_properties(${target} PROPERTIES TIMEOUT 60)
        set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
      endif()
    else()
      py_test_modules(${target} MODULES ${target})
      set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
    endif()
  endforeach()

  foreach(target ${TEST_TRT_CONVERTER})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES TIMEOUT 300)
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

file(
  GLOB TEST_MKLDNN_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_mkldnn_*.py")
string(REPLACE ".py" "" TEST_MKLDNN_IR_PASSES "${TEST_MKLDNN_IR_PASSES}")
foreach(TEST_INFERENCE_IR_PASS ${TEST_MKLDNN_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

file(
  GLOB TEST_ONEDNN_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_onednn_*.py")
string(REPLACE ".py" "" TEST_ONEDNN_IR_PASSES "${TEST_ONEDNN_IR_PASSES}")
foreach(TEST_INFERENCE_IR_PASS ${TEST_ONEDNN_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_INFERENCE_IR_PASS})
endforeach()

if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
  message(STATUS "Skip tests unrelated to CUDA/TRT")
elseif(WITH_ONEDNN)
  foreach(target ${TEST_MKLDNN_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()

  foreach(target ${TEST_ONEDNN_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

file(
  GLOB TEST_XPU_IR_PASSES
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_xpu_*.py")
string(REPLACE ".py" "" TEST_XPU_IR_PASSES "${TEST_XPU_IR_PASSES}")
foreach(TEST_XPU_IR_PASS ${TEST_XPU_IR_PASSES})
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES ${TEST_XPU_IR_PASS})
endforeach()

if(WITH_XPU)
  foreach(target ${TEST_XPU_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

if(WITH_ONEDNN
   AND TENSORRT_FOUND
   AND WITH_GPU)
  foreach(target ${TEST_INFERENCE_IR_PASSES})
    py_test_modules(${target} MODULES ${target})
    set_tests_properties(${target} PROPERTIES LABELS "RUN_TYPE=INFER")
  endforeach()
endif()

if(WITH_GPU AND TENSORRT_FOUND)
  # longer timeout for trt_activation_pass for longer trt optimization time in trt 8
  set_tests_properties(test_trt_conv_pass PROPERTIES TIMEOUT 120)
  #set_tests_properties(test_trt_multiclass_nms_op PROPERTIES TIMEOUT 200)
  set_tests_properties(test_trt_dynamic_shape PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_inference_predictor PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_inference_fp16_io PROPERTIES TIMEOUT 500)
  if(NOT WIN32)
    set_tests_properties(test_trt_explicit_quantization_mobilenet
                         PROPERTIES TIMEOUT 300)
  else()
    set_tests_properties(test_trt_convert_fill_constant PROPERTIES TIMEOUT 450)
  endif()

  set_tests_properties(test_trt_tile_op PROPERTIES TIMEOUT 60)

  set_tests_properties(test_trt_conv3d_op PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_conv3d_transpose_op PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_nearest_interp_v2_op PROPERTIES TIMEOUT 30)
  set_tests_properties(test_trt_multiclass_nms3_op PROPERTIES TIMEOUT 60)

  if(WITH_ONEDNN)

    set_tests_properties(test_conv_transpose_eltwiseadd_bn_fuse_pass
                         PROPERTIES TIMEOUT 250)
    if(WIN32)
      set_tests_properties(test_layernorm_shift_partition_pass
                           PROPERTIES TIMEOUT 360)
      if(WIN_UNITTEST_LEVEL EQUAL 2)
        set_tests_properties(test_onednn_conv_bias_fuse_pass PROPERTIES TIMEOUT
                                                                        300)
        set_tests_properties(test_onednn_conv_concat_activation_fuse_pass
                             PROPERTIES TIMEOUT 300)
        set_tests_properties(test_onednn_multi_gru_fuse_pass PROPERTIES TIMEOUT
                                                                        120)
        set_tests_properties(test_onednn_multi_gru_seq_fuse_pass
                             PROPERTIES TIMEOUT 120)
      endif()
    endif()
    if(WIN32 AND WIN_UNITTEST_LEVEL LESS 2)
      message(STATUS "Skip tests unrelated to CUDA/TRT")
    else()
      set_tests_properties(test_mkldnn_depthwise_conv_pass PROPERTIES TIMEOUT
                                                                      120)

      set_tests_properties(test_conv_transpose_bn_fuse_pass PROPERTIES TIMEOUT
                                                                       300)
    endif()
  endif()
endif()
