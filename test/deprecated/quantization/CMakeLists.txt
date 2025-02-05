file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(_inference_analysis_python_api_int8_test target model_dir data_path
         filename use_mkldnn)
  py_test(
    ${target}
    SRCS ${filename}
         ENVS
         CPU_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         FLAGS_use_mkldnn=${use_mkldnn}
         ARGS
         --infer_model
         ${model_dir}/model
         --infer_data
         ${data_path}
         --int8_model_save_path
         int8_models/${target}
         --warmup_batch_size
         ${WARMUP_BATCH_SIZE}
         --batch_size
         50)
endfunction()

function(inference_analysis_python_api_int8_test target model_dir data_path
         filename)
  _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path}
                                           ${filename} False)
endfunction()

function(inference_analysis_python_api_int8_test_custom_warmup_batch_size
         target model_dir data_dir filename warmup_batch_size)
  set(WARMUP_BATCH_SIZE ${warmup_batch_size})
  inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_dir}
                                          ${filename})
endfunction()

function(inference_analysis_python_api_int8_test_mkldnn target model_dir
         data_path filename)
  _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path}
                                           ${filename} True)
endfunction()

function(download_quant_data install_dir data_file check_sum)
  if(NOT EXISTS ${install_dir}/${data_file})
    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8
                                      ${data_file} ${check_sum})
  endif()
endfunction()

function(download_quant_fp32_model install_dir data_file check_sum)
  if(NOT EXISTS ${install_dir}/${data_file})
    inference_download_and_uncompress(
      ${install_dir} ${INFERENCE_URL}/int8/QAT_models/fp32 ${data_file}
      ${check_sum})
  endif()
endfunction()

function(inference_quant_int8_image_classification_test target quant_model_dir
         dataset_path)
  py_test(
    ${target}
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant_int8_image_classification_comparison.py"
         ENVS
         FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         FLAGS_use_mkldnn=true
         ARGS
         --quant_model
         ${quant_model_dir}
         --infer_data
         ${dataset_path}
         --batch_size
         25
         --batch_num
         2
         --acc_diff_threshold
         0.1)
endfunction()

# set batch_size 10 for UT only (avoid OOM).
# For whole dataset, use batch_size 25
function(inference_quant2_int8_image_classification_test target quant_model_dir
         fp32_model_dir dataset_path)
  py_test(
    ${target}
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant2_int8_image_classification_comparison.py"
         ENVS
         FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         FLAGS_use_mkldnn=true
         ARGS
         --quant_model
         ${quant_model_dir}
         --fp32_model
         ${fp32_model_dir}
         --infer_data
         ${dataset_path}
         --batch_size
         50
         --batch_num
         2
         --acc_diff_threshold
         0.1)
endfunction()

# set batch_size 10 for UT only (avoid OOM).
# For whole dataset, use batch_size 20
function(
  inference_quant2_int8_nlp_test
  target
  quant_model_dir
  fp32_model_dir
  dataset_path
  labels_path
  ops_to_quantize)
  py_test(
    ${target}
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant2_int8_nlp_comparison.py"
         ENVS
         FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
         FLAGS_use_mkldnn=true
         ARGS
         --quant_model
         ${quant_model_dir}
         --fp32_model
         ${fp32_model_dir}
         --infer_data
         ${dataset_path}
         --labels
         ${labels_path}
         --batch_size
         10
         --batch_num
         2
         --acc_diff_threshold
         0.1
         --ops_to_quantize
         ${ops_to_quantize})
endfunction()

function(inference_quant2_int8_lstm_model_test target fp32_model quant_model
         dataset_path)
  py_test(
    ${target}
    SRCS "${CMAKE_CURRENT_SOURCE_DIR}/quant2_int8_lstm_model.py"
         ARGS
         --fp32_model
         ${fp32_model}
         --quant_model
         ${quant_model}
         --infer_data
         ${dataset_path}
         --num_threads
         1
         --mkldnn_cache_capacity
         100
         --warmup_iter
         100
         --acc_diff_threshold
         0.11)
endfunction()

function(download_quant_data install_dir data_file check_sum)
  if(NOT EXISTS ${install_dir}/${data_file})
    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8
                                      ${data_file} ${check_sum})
  endif()
endfunction()

function(convert_model2dot_test target model_path save_graph_dir
         save_graph_name)
  py_test(
    ${target}
    SRCS ${CMAKE_CURRENT_SOURCE_DIR}/convert_model2dot.py
         ARGS
         --model_path
         ${model_path}
         --save_graph_dir
         ${save_graph_dir}
         --save_graph_name
         ${save_graph_name})
endfunction()

if(WIN32)
  list(REMOVE_ITEM TEST_OPS test_light_nas)
  list(REMOVE_ITEM TEST_OPS test_post_training_quantization_while_deprecated)
  list(REMOVE_ITEM TEST_OPS test_weight_quantization_mobilenetv1_deprecated)
  list(REMOVE_ITEM TEST_OPS test_imperative_qat_amp)
  list(REMOVE_ITEM TEST_OPS test_weight_only_linear)
  list(REMOVE_ITEM TEST_OPS test_llm_int8_linear)
  list(REMOVE_ITEM TEST_OPS test_quant_aware_deprecated)
  list(REMOVE_ITEM TEST_OPS test_quant_post_quant_aware_deprecated)
  list(REMOVE_ITEM TEST_OPS test_quant_aware_user_defined_deprecated)
  list(REMOVE_ITEM TEST_OPS test_quant_amp_deprecated)
  list(REMOVE_ITEM TEST_OPS test_apply_per_channel_scale)

endif()

if(NOT WITH_GPU)
  list(REMOVE_ITEM TEST_OPS test_weight_only_linear)
  list(REMOVE_ITEM TEST_OPS test_llm_int8_linear)
  list(REMOVE_ITEM TEST_OPS test_apply_per_channel_scale)
endif()

if(LINUX AND WITH_MKLDNN)

  #### Image classification dataset: ImageNet (small)
  # The dataset should already be downloaded for INT8v2 unit tests
  set(IMAGENET_DATA_PATH "${INFERENCE_DEMO_INSTALL_DIR}/imagenet/data.bin")

  #### INT8 image classification python api test
  # Models should be already downloaded for INT8v2 unit tests

  set(INT8_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")

  #### QUANT & INT8 comparison python api tests

  set(QUANT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/quant")

endif()

# Since the tests for Quant & INT8 comparison support only testing on Linux
# with MKL-DNN, we remove it here to not test it on other systems.
list(REMOVE_ITEM TEST_OPS test_mkldnn_int8_quantization_strategy
     quant_int8_image_classification_comparison quant_int8_nlp_comparison)

#TODO(wanghaoshuang): Fix this unittest failed on GCC8.
list(REMOVE_ITEM TEST_OPS test_auto_pruning)
list(REMOVE_ITEM TEST_OPS test_filter_pruning)

# fix
if(WIN32)
  set(SINGLE_CARD_TEST_OPS
      test_user_defined_quantization_deprecated
      test_quantization_scale_pass_deprecated test_quantization_pass_deprecated
      test_moving_average_abs_max_scale_op_deprecated test_graph_deprecated)
  list(REMOVE_ITEM TEST_OPS ${SINGLE_CARD_TEST_OPS})
  foreach(src ${SINGLE_CARD_TEST_OPS})
    py_test(${src} SRCS ${src}.py ENVS CUDA_VISIBLE_DEVICES=0)
  endforeach()
endif()

foreach(src ${TEST_OPS})
  py_test(${src} SRCS ${src}.py)
endforeach()

# setting timeout value for old unittests
if(NOT WIN32)
  set_tests_properties(test_post_training_quantization_while_deprecated
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_weight_quantization_mobilenetv1_deprecated
                       PROPERTIES TIMEOUT 120)
  set_tests_properties(test_quant_aware_deprecated PROPERTIES TIMEOUT 200)
  set_tests_properties(test_quant_post_quant_aware_deprecated PROPERTIES TIMEOUT
                                                                         200)
  set_tests_properties(test_quant_aware_user_defined_deprecated
                       PROPERTIES TIMEOUT 200)
  set_tests_properties(test_quant_amp_deprecated PROPERTIES TIMEOUT 200)
endif()

set_tests_properties(test_graph_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_quantization_pass_deprecated PROPERTIES TIMEOUT 120)
set_tests_properties(test_user_defined_quantization_deprecated
                     PROPERTIES TIMEOUT 200)

if(APPLE)
  set_tests_properties(test_post_training_quantization_while_deprecated
                       PROPERTIES TIMEOUT 300)
endif()

set_tests_properties(test_quantization_scale_pass_deprecated PROPERTIES TIMEOUT
                                                                        100)
