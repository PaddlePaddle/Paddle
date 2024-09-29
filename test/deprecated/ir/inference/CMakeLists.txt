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

list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_inspector_deprecated")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_trt_inspector_deprecated")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_trt_convert_temporal_shift_deprecated")
list(REMOVE_ITEM TEST_TRT_IR_PASSES
     "test_trt_convert_temporal_shift_deprecated")
list(REMOVE_ITEM TEST_TRT_CONVERTER
     "test_trt_convert_temporal_shift_deprecated")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_trt_convert_pad3d_deprecated")
list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_convert_pad3d_deprecated")
list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_pad3d_deprecated")
list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_multiclass_nms3_op_deprecated")
list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
     "test_trt_multiclass_nms3_op_deprecated")

if(WIN32)
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_inference_fp16_io_deprecated")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_inference_fp16_io_deprecated")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_convert_depthwise_conv2d_transpose_deprecated")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES
       "test_trt_convert_depthwise_conv2d_transpose_deprecated")
  list(REMOVE_ITEM TEST_TRT_CONVERTER
       "test_trt_convert_depthwise_conv2d_transpose_deprecated")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_convert_conv2d_deprecated")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_convert_conv2d_deprecated")
  list(REMOVE_ITEM TEST_TRT_CONVERTER "test_trt_convert_conv2d_deprecated")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES "test_trt_pool3d_op_deprecated")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_pool3d_op_deprecated")
  list(REMOVE_ITEM TEST_INFERENCE_IR_PASSES
       "test_trt_deformable_conv_deprecated")
  list(REMOVE_ITEM TEST_TRT_IR_PASSES "test_trt_deformable_conv_deprecated")

endif()

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

if(WITH_GPU AND TENSORRT_FOUND)
  set_tests_properties(test_trt_subgraph_pass_deprecated PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_conv_pass_deprecated PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_dynamic_shape_deprecated PROPERTIES TIMEOUT 120)
  set_tests_properties(test_trt_inference_predictor_deprecated
                       PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_optimization_level_deprecated PROPERTIES TIMEOUT
                                                                         300)
  set_tests_properties(test_trt_elementwise_op_deprecated PROPERTIES TIMEOUT
                                                                     300)
  set_tests_properties(test_trt_fc_fuse_pass_deprecated PROPERTIES TIMEOUT 500)
  set_tests_properties(test_trt_flatten_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_gather_nd_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_gather_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_instance_norm_op_deprecated PROPERTIES TIMEOUT
                                                                       300)
  set_tests_properties(test_trt_matmul_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_nearest_interp_op_deprecated PROPERTIES TIMEOUT
                                                                        300)
  set_tests_properties(test_trt_ops_fp16_mix_precision_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_pad_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_reduce_sum_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_reshape_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_scale_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_shuffle_channel_detect_pass_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_skip_layernorm_fuse_pass_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_slice_dynamic_plugin_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_support_nhwc_pass_deprecated PROPERTIES TIMEOUT
                                                                        300)
  set_tests_properties(test_trt_transpose_flatten_concat_fuse_pass_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_tuned_dynamic_shape_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_while_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_yolo_box_op_deprecated PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_convert_conv2d_transpose_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_convert_conv3d_transpose_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_convert_depthwise_conv2d_deprecated
                       PROPERTIES TIMEOUT 300)
  if(NOT WIN32)
    set_tests_properties(test_trt_inference_fp16_io_deprecated
                         PROPERTIES TIMEOUT 500)
    set_tests_properties(
      test_trt_pool3d_op_deprecated
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 450)
    set_tests_properties(test_trt_deformable_conv_deprecated PROPERTIES TIMEOUT
                                                                        500)
    set_tests_properties(test_trt_convert_conv2d_deprecated PROPERTIES TIMEOUT
                                                                       500)
    set_tests_properties(test_trt_convert_depthwise_conv2d_transpose_deprecated
                         PROPERTIES TIMEOUT 500)

  endif()
  if(WITH_NV_JETSON)
    set_tests_properties(
      test_trt_pool_op_deprecated
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 550)
    set_tests_properties(
      test_trt_pool3d_op_deprecated
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 550)
  else()
    set_tests_properties(
      test_trt_pool_op_deprecated
      PROPERTIES ENVIRONMENT FLAGS_fraction_of_gpu_memory_to_use=0.1 TIMEOUT
                 500)
  endif()

  set_tests_properties(test_trt_tile_op_deprecated PROPERTIES TIMEOUT 60)
  set_tests_properties(test_trt_fc_fuse_quant_dequant_pass_deprecated
                       PROPERTIES TIMEOUT 300)
  set_tests_properties(test_trt_conv_quant_dequant_pass_deprecated
                       PROPERTIES TIMEOUT 500)
  set_tests_properties(test_trt_matmul_quant_dequant_deprecated
                       PROPERTIES TIMEOUT 500)
  set_tests_properties(test_trt_conv3d_transpose_op_deprecated
                       PROPERTIES TIMEOUT 500)
  set_tests_properties(test_trt_nearest_interp_v2_op_deprecated
                       PROPERTIES TIMEOUT 500)
endif()
