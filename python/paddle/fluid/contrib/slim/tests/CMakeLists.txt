file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(inference_analysis_python_api_int8_test target model_dir data_dir filename)
    py_test(${target} SRCS ${filename}
        ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
        ARGS --infer_model ${model_dir}/model
             --infer_data ${data_dir}/data.bin
             --int8_model_save_path int8_models/${target}
             --warmup_batch_size 100
             --batch_size 50)
endfunction()

# NOTE: TODOOOOOOOOOOO
# temporarily disable test_distillation_strategy since it always failed on a specified machine with 4 GPUs
# Need to figure out the root cause and then add it back
list(REMOVE_ITEM TEST_OPS test_distillation_strategy)

if(WIN32)
    list(REMOVE_ITEM TEST_OPS test_light_nas)
endif()

# int8 image classification python api test
if(LINUX AND WITH_MKLDNN)
  set(INT8_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")
  set(MKLDNN_INT8_TEST_FILE "test_mkldnn_int8_quantization_strategy.py")

  # googlenet int8
  set(INT8_GOOGLENET_MODEL_DIR "${INT8_DATA_DIR}/googlenet")
  inference_analysis_python_api_int8_test(test_slim_int8_googlenet ${INT8_GOOGLENET_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

  # mobilenet int8
  set(INT8_MOBILENET_MODEL_DIR "${INT8_DATA_DIR}/mobilenet")
  inference_analysis_python_api_int8_test(test_slim_int8_mobilenet ${INT8_MOBILENET_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

  # temporarily adding WITH_SLIM_MKLDNN_FULL_TEST FLAG for QA testing the following UTs locally, 
  # since the following UTs cost too much time on CI test.
  if (WITH_SLIM_MKLDNN_FULL_TEST)
    # resnet50 int8
    set(INT8_RESNET50_MODEL_DIR "${INT8_DATA_DIR}/resnet50")
    inference_analysis_python_api_int8_test(test_slim_int8_resnet50 ${INT8_RESNET50_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

    # mobilenetv2 int8
    set(INT8_MOBILENETV2_MODEL_DIR "${INT8_DATA_DIR}/mobilenetv2")
    inference_analysis_python_api_int8_test(test_slim_int8_mobilenetv2 ${INT8_MOBILENETV2_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

    # resnet101 int8
    set(INT8_RESNET101_MODEL_DIR "${INT8_DATA_DIR}/resnet101")
    inference_analysis_python_api_int8_test(test_slim_int8_resnet101 ${INT8_RESNET101_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

    # vgg16 int8
    set(INT8_VGG16_MODEL_DIR "${INT8_DATA_DIR}/vgg16")
    inference_analysis_python_api_int8_test(test_slim_int8_vgg16 ${INT8_VGG16_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})

    # vgg19 int8
    set(INT8_VGG19_MODEL_DIR "${INT8_DATA_DIR}/vgg19")
    inference_analysis_python_api_int8_test(test_slim_int8_vgg19 ${INT8_VGG19_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_INT8_TEST_FILE})
  endif()
endif()

# Since test_mkldnn_int8_quantization_strategy only supports testing on Linux 
# with MKL-DNN, we remove it here for not repeating test, or not testing on other systems.
list(REMOVE_ITEM TEST_OPS test_mkldnn_int8_quantization_strategy)

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
