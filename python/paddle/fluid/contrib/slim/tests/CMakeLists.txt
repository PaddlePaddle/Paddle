file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(inference_qat_int8_test target model_dir data_dir test_script use_mkldnn)
    py_test(${target} SRCS ${test_script}
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
	    	 FLAGS_use_mkldnn=${use_mkldnn}
            ARGS --qat_model ${model_dir}/model
                 --infer_data ${data_dir}/data.bin
		 --batch_size 50
		 --batch_num 2
		 --acc_diff_threshold 0.01
		 --out_diff_threshold 1e-05)
endfunction()

# NOTE: TODOOOOOOOOOOO
# temporarily disable test_distillation_strategy since it always failed on a specified machine with 4 GPUs
# Need to figure out the root cause and then add it back
list(REMOVE_ITEM TEST_OPS test_distillation_strategy)

# QAT vs INT8v2 after QAT comparison python api tests
if(LINUX AND WITH_MKLDNN)
	set(INT8_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")
	if (NOT EXISTS ${INT8_DATA_DIR})
		inference_download_and_uncompress(${INT8_DATA_DIR} "${INFERENCE_URL}/int8" "imagenet_val_100_tail.tar.gz")
	endif()

	set(QAT_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/qat")
	set(MKLDNN_QAT_TEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/qat_int8_comparison.py")

	# ResNet50
	set(QAT_RESNET50_MODEL_DIR "${QAT_DATA_DIR}/ResNet50")
	if (NOT EXISTS ${QAT_RESNET50_MODEL_DIR})
		inference_download_and_uncompress(${QAT_RESNET50_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "ResNet50_qat_model.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_resnet50 ${QAT_RESNET50_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_resnet50_mkldnn ${QAT_RESNET50_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# ResNet101
	set(QAT_RESNET101_MODEL_DIR "${QAT_DATA_DIR}/ResNet101")
	if (NOT EXISTS ${QAT_RESNET101_MODEL_DIR})
		inference_download_and_uncompress(${QAT_RESNET101_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "ResNet101_qat_model.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_resnet101 ${QAT_RESNET101_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_resnet101_mkldnn ${QAT_RESNET101_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# GoogleNet
	set(QAT_GOOGLENET_MODEL_DIR "${QAT_DATA_DIR}/GoogleNet")
	if (NOT EXISTS ${QAT_GOOGLENET_MODEL_DIR})
		inference_download_and_uncompress(${QAT_GOOGLENET_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "GoogleNet_quant.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_googlenet ${QAT_GOOGLENET_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_googlenet_mkldnn ${QAT_GOOGLENET_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# MobileNetV1
	set(QAT_MOBILENETV1_MODEL_DIR "${QAT_DATA_DIR}/MobileNetV1")
	if (NOT EXISTS ${QAT_MOBILENETV1_MODEL_DIR})
		inference_download_and_uncompress(${QAT_MOBILENETV1_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "MobileNetV1_quant.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_mobilenetv1 ${QAT_MOBILENETV1_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_mobilenetv1_mkldnn ${QAT_MOBILENETV1_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# MobileNetV2
	set(QAT_MOBILENETV2_MODEL_DIR "${QAT_DATA_DIR}/MobileNetV2")
	if (NOT EXISTS ${QAT_MOBILENETV2_MODEL_DIR})
		inference_download_and_uncompress(${QAT_MOBILENETV2_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "MobileNetV2_quant.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_mobilenetv2 ${QAT_MOBILENETV2_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_mobilenetv2_mkldnn ${QAT_MOBILENETV2_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# VGG16
	set(QAT_VGG16_MODEL_DIR "${QAT_DATA_DIR}/VGG16")
	if (NOT EXISTS ${QAT_VGG16_MODEL_DIR})
		inference_download_and_uncompress(${QAT_VGG16_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "VGG16_quant.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_vgg16 ${QAT_VGG16_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_vgg16_mkldnn ${QAT_VGG16_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

	# VGG19
	set(QAT_VGG19_MODEL_DIR "${QAT_DATA_DIR}/VGG19")
	if (NOT EXISTS ${QAT_VGG19_MODEL_DIR})
		inference_download_and_uncompress(${QAT_VGG19_MODEL_DIR} "${INFERENCE_URL}/int8/qat" "VGG19_quant.tar.gz" )
	endif()
	inference_qat_int8_test(test_qat_int8_vgg19 ${QAT_VGG19_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} false)
	inference_qat_int8_test(test_qat_int8_vgg19_mkldnn ${QAT_VGG19_MODEL_DIR} ${INT8_DATA_DIR} ${MKLDNN_QAT_TEST_FILE} true)

endif()

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
