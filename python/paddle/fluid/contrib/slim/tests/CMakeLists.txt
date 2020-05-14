file(GLOB TEST_OPS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

function(_inference_analysis_python_api_int8_test target model_dir data_path filename use_mkldnn)
    py_test(${target} SRCS ${filename}
        ENVS CPU_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
             FLAGS_use_mkldnn=${use_mkldnn}
        ARGS --infer_model ${model_dir}/model
             --infer_data ${data_path}
             --int8_model_save_path int8_models/${target}
             --warmup_batch_size ${WARMUP_BATCH_SIZE}
             --batch_size 50)
endfunction()

function(inference_analysis_python_api_int8_test target model_dir data_path filename)
    _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path} ${filename} False)
endfunction()

function(inference_analysis_python_api_int8_test_custom_warmup_batch_size target model_dir data_dir filename warmup_batch_size)
    set(WARMUP_BATCH_SIZE ${warmup_batch_size})
    inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_dir} ${filename})
endfunction()

function(inference_analysis_python_api_int8_test_mkldnn target model_dir data_path filename)
    _inference_analysis_python_api_int8_test(${target} ${model_dir} ${data_path} ${filename} True)
endfunction()

function(download_qat_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8 ${data_file})
    endif()
endfunction()

function(download_qat_model install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file})
    endif()
endfunction()

function(download_qat_fp32_model install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
	    inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models/fp32 ${data_file})
    endif()
endfunction()

function(inference_qat_int8_image_classification_test target qat_model_dir dataset_path)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/qat_int8_image_classification_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --qat_model ${qat_model_dir}
                 --infer_data ${dataset_path}
                 --batch_size 25
                 --batch_num 2
                 --acc_diff_threshold 0.1)
endfunction()


# set batch_size 10 for UT only (avoid OOM). For whole dataset, use batch_size 25 
function(inference_qat2_int8_image_classification_test target qat_model_dir fp32_model_dir dataset_path ops_to_quantize)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/qat2_int8_image_classification_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --qat_model ${qat_model_dir}
                 --fp32_model ${fp32_model_dir}
                 --infer_data ${dataset_path}
                 --batch_size 10
                 --batch_num 2
                 --acc_diff_threshold 0.1
                 --ops_to_quantize ${ops_to_quantize})
endfunction()

# set batch_size 10 for UT only (avoid OOM). For whole dataset, use batch_size 20 
function(inference_qat2_int8_nlp_test target qat_model_dir fp32_model_dir dataset_path labels_path)
    py_test(${target} SRCS "${CMAKE_CURRENT_SOURCE_DIR}/qat2_int8_nlp_comparison.py"
            ENVS FLAGS_OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 OMP_NUM_THREADS=${CPU_NUM_THREADS_ON_CI}
                 FLAGS_use_mkldnn=true
            ARGS --qat_model ${qat_model_dir}
		 --fp32_model ${fp32_model_dir}
                 --infer_data ${dataset_path}
		 --labels ${labels_path}
                 --batch_size 10
                 --batch_num 2
                 --acc_diff_threshold 0.1)
endfunction()

function(download_qat_data install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
           inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8 ${data_file})
    endif()
endfunction()

function(download_qat_model install_dir data_file)
    if (NOT EXISTS ${install_dir}/${data_file})
           inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file})
    endif()
endfunction()

function(save_qat_ic_model_test target qat_model_dir fp32_model_save_path int8_model_save_path ops_to_quantize)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/save_qat_model.py
            ARGS --qat_model_path ${qat_model_dir}
	         --fp32_model_save_path ${fp32_model_save_path}
	         --int8_model_save_path ${int8_model_save_path}
		 --ops_to_quantize ${ops_to_quantize})
endfunction()

function(save_qat_nlp_model_test target qat_model_dir fp32_model_save_path int8_model_save_path)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/save_qat_model.py
            ARGS --qat_model_path ${qat_model_dir}
	         --fp32_model_save_path ${fp32_model_save_path}
	         --int8_model_save_path ${int8_model_save_path})
endfunction()

function(convert_model2dot_test target model_path save_graph_dir save_graph_name)
    py_test(${target} SRCS ${CMAKE_CURRENT_SOURCE_DIR}/convert_model2dot.py
            ARGS --model_path ${model_path}
	         --save_graph_dir ${save_graph_dir}
	         --save_graph_name ${save_graph_name})
endfunction()

if(WIN32)
	list(REMOVE_ITEM TEST_OPS test_light_nas)
	list(REMOVE_ITEM TEST_OPS test_post_training_quantization_mobilenetv1)
    list(REMOVE_ITEM TEST_OPS test_post_training_quantization_resnet50)
    list(REMOVE_ITEM TEST_OPS test_weight_quantization_mobilenetv1)
endif()

if(LINUX AND WITH_MKLDNN)

	#### Image classification dataset: ImageNet (small)
	# The dataset should already be downloaded for INT8v2 unit tests
	set(IMAGENET_DATA_PATH "${INFERENCE_DEMO_INSTALL_DIR}/imagenet/data.bin")

	#### INT8 image classification python api test
	# Models should be already downloaded for INT8v2 unit tests

	set(INT8_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/int8v2")
	set(INT8_IC_TEST_FILE "test_mkldnn_int8_quantization_strategy.py")
	set(INT8_IC_TEST_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${INT8_IC_TEST_FILE}")

	# googlenet int8
	set(INT8_GOOGLENET_MODEL_DIR "${INT8_INSTALL_DIR}/googlenet")
	inference_analysis_python_api_int8_test_custom_warmup_batch_size(test_slim_int8_googlenet ${INT8_GOOGLENET_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH} 10)

	# mobilenet int8
	set(INT8_MOBILENET_MODEL_DIR "${INT8_INSTALL_DIR}/mobilenetv1")
	inference_analysis_python_api_int8_test(test_slim_int8_mobilenet ${INT8_MOBILENET_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})
	inference_analysis_python_api_int8_test_mkldnn(test_slim_int8_mobilenet_mkldnn ${INT8_MOBILENET_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})

	# temporarily adding WITH_SLIM_MKLDNN_FULL_TEST FLAG for QA testing the following UTs locally,
	# since the following UTs cost too much time on CI test.
	if (WITH_SLIM_MKLDNN_FULL_TEST)
		# resnet50 int8
		set(INT8_RESNET50_MODEL_DIR "${INT8_INSTALL_DIR}/resnet50")
		inference_analysis_python_api_int8_test(test_slim_int8_resnet50 ${INT8_RESNET50_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})

		# mobilenetv2 int8
		set(INT8_MOBILENETV2_MODEL_DIR "${INT8_INSTALL_DIR}/mobilenetv2")
		inference_analysis_python_api_int8_test(test_slim_int8_mobilenetv2 ${INT8_MOBILENETV2_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})

		# resnet101 int8
		set(INT8_RESNET101_MODEL_DIR "${INT8_INSTALL_DIR}/resnet101")
		inference_analysis_python_api_int8_test(test_slim_int8_resnet101 ${INT8_RESNET101_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})

		# vgg16 int8
		set(INT8_VGG16_MODEL_DIR "${INT8_INSTALL_DIR}/vgg16")
		inference_analysis_python_api_int8_test(test_slim_int8_vgg16 ${INT8_VGG16_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})

		# vgg19 int8
		set(INT8_VGG19_MODEL_DIR "${INT8_INSTALL_DIR}/vgg19")
		inference_analysis_python_api_int8_test(test_slim_int8_vgg19 ${INT8_VGG19_MODEL_DIR} ${IMAGENET_DATA_PATH} ${INT8_IC_TEST_FILE_PATH})
	endif()

	#### QAT FP32 & INT8 comparison python api tests

	set(QAT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/qat")

	### QATv1 for image classification

	# QAT ResNet50
	set(QAT_RESNET50_MODEL_DIR "${QAT_INSTALL_DIR}/ResNet50_QAT")
	set(QAT_RESNET50_MODEL_ARCHIVE "ResNet50_qat_model.tar.gz")
	download_qat_model(${QAT_RESNET50_MODEL_DIR} ${QAT_RESNET50_MODEL_ARCHIVE})
	inference_qat_int8_image_classification_test(test_qat_int8_resnet50_mkldnn ${QAT_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT ResNet101
	set(QAT_RESNET101_MODEL_DIR "${QAT_INSTALL_DIR}/ResNet101_QAT")
	set(QAT_RESNET101_MODEL_ARCHIVE "ResNet101_qat_model.tar.gz")
	download_qat_model(${QAT_RESNET101_MODEL_DIR} ${QAT_RESNET101_MODEL_ARCHIVE})
	# inference_qat_int8_image_classification_test(test_qat_int8_resnet101_mkldnn ${QAT_RESNET101_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT GoogleNet
	set(QAT_GOOGLENET_MODEL_DIR "${QAT_INSTALL_DIR}/GoogleNet_QAT")
	set(QAT_GOOGLENET_MODEL_ARCHIVE "GoogleNet_qat_model.tar.gz")
	download_qat_model(${QAT_GOOGLENET_MODEL_DIR} ${QAT_GOOGLENET_MODEL_ARCHIVE})
	inference_qat_int8_image_classification_test(test_qat_int8_googlenet_mkldnn ${QAT_GOOGLENET_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT MobileNetV1
	set(QAT_MOBILENETV1_MODEL_DIR "${QAT_INSTALL_DIR}/MobileNetV1_QAT")
	set(QAT_MOBILENETV1_MODEL_ARCHIVE "MobileNetV1_qat_model.tar.gz")
	download_qat_model(${QAT_MOBILENETV1_MODEL_DIR} ${QAT_MOBILENETV1_MODEL_ARCHIVE})
	inference_qat_int8_image_classification_test(test_qat_int8_mobilenetv1_mkldnn ${QAT_MOBILENETV1_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT MobileNetV2
	set(QAT_MOBILENETV2_MODEL_DIR "${QAT_INSTALL_DIR}/MobileNetV2_QAT")
	set(QAT_MOBILENETV2_MODEL_ARCHIVE "MobileNetV2_qat_model.tar.gz")
	download_qat_model(${QAT_MOBILENETV2_MODEL_DIR} ${QAT_MOBILENETV2_MODEL_ARCHIVE})
	inference_qat_int8_image_classification_test(test_qat_int8_mobilenetv2_mkldnn ${QAT_MOBILENETV2_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT VGG16
	set(QAT_VGG16_MODEL_DIR "${QAT_INSTALL_DIR}/VGG16_QAT")
	set(QAT_VGG16_MODEL_ARCHIVE "VGG16_qat_model.tar.gz")
	download_qat_model(${QAT_VGG16_MODEL_DIR} ${QAT_VGG16_MODEL_ARCHIVE})
	# inference_qat_int8_image_classification_test(test_qat_int8_vgg16_mkldnn ${QAT_VGG16_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	# QAT VGG19
	set(QAT_VGG19_MODEL_DIR "${QAT_INSTALL_DIR}/VGG19_QAT")
	set(QAT_VGG19_MODEL_ARCHIVE "VGG19_qat_model.tar.gz")
	download_qat_model(${QAT_VGG19_MODEL_DIR} ${QAT_VGG19_MODEL_ARCHIVE})
	# inference_qat_int8_image_classification_test(test_qat_int8_vgg19_mkldnn ${QAT_VGG19_MODEL_DIR}/model ${IMAGENET_DATA_PATH})

	### QATv2 for image classification

	set(QAT2_IC_OPS_TO_QUANTIZE "conv2d,pool2d")

	# QAT2 ResNet50 with input/output scales in `fake_quantize_moving_average_abs_max` operators,
	# with weight scales in `fake_dequantize_max_abs` operators
        set(QAT2_RESNET50_MODEL_DIR "${QAT_INSTALL_DIR}/ResNet50_qat_perf")
	set(FP32_RESNET50_MODEL_DIR "${INT8_INSTALL_DIR}/resnet50")
	set(QAT2_RESNET50_MODEL_ARCHIVE "ResNet50_qat_perf.tar.gz")
	download_qat_model(${QAT2_RESNET50_MODEL_DIR} ${QAT2_RESNET50_MODEL_ARCHIVE})
	inference_qat2_int8_image_classification_test(test_qat2_int8_resnet50_mkldnn ${QAT2_RESNET50_MODEL_DIR}/ResNet50_qat_perf/float ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH} ${QAT2_IC_OPS_TO_QUANTIZE})

	# QAT2 ResNet50 with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes,
	# with weight scales in `fake_dequantize_max_abs` operators
	set(QAT2_RESNET50_RANGE_MODEL_DIR "${QAT_INSTALL_DIR}/ResNet50_qat_range")
	set(QAT2_RESNET50_RANGE_MODEL_ARCHIVE "ResNet50_qat_range.tar.gz")
	download_qat_model(${QAT2_RESNET50_RANGE_MODEL_DIR} ${QAT2_RESNET50_RANGE_MODEL_ARCHIVE})
	inference_qat2_int8_image_classification_test(test_qat2_int8_resnet50_range_mkldnn ${QAT2_RESNET50_RANGE_MODEL_DIR}/ResNet50_qat_range ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH} ${QAT2_IC_OPS_TO_QUANTIZE})

	# QAT2 ResNet50 with input/output scales in `fake_quantize_range_abs_max` operators and the `out_threshold` attributes,
	# with weight scales in `fake_channel_wise_dequantize_max_abs` operators
	set(QAT2_RESNET50_CHANNELWISE_MODEL_DIR "${QAT_INSTALL_DIR}/ResNet50_qat_channelwise")
	set(QAT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE "ResNet50_qat_channelwise.tar.gz")
	download_qat_model(${QAT2_RESNET50_CHANNELWISE_MODEL_DIR} ${QAT2_RESNET50_CHANNELWISE_MODEL_ARCHIVE})
	inference_qat2_int8_image_classification_test(test_qat2_int8_resnet50_channelwise_mkldnn ${QAT2_RESNET50_CHANNELWISE_MODEL_DIR}/ResNet50_qat_channelwise ${FP32_RESNET50_MODEL_DIR}/model ${IMAGENET_DATA_PATH} ${QAT2_IC_OPS_TO_QUANTIZE})

	# QAT2 MobileNetV1
        set(QAT2_MOBILENETV1_MODEL_DIR "${QAT_INSTALL_DIR}/MobileNet_qat_perf")
	set(FP32_MOBILENETV1_MODEL_DIR "${INT8_INSTALL_DIR}/mobilenetv1")
	set(QAT2_MOBILENETV1_MODEL_ARCHIVE "MobileNet_qat_perf.tar.gz")
	download_qat_model(${QAT2_MOBILENETV1_MODEL_DIR} ${QAT2_MOBILENETV1_MODEL_ARCHIVE})
	inference_qat2_int8_image_classification_test(test_qat2_int8_mobilenetv1_mkldnn ${QAT2_MOBILENETV1_MODEL_DIR}/MobileNet_qat_perf/float ${FP32_MOBILENETV1_MODEL_DIR}/model ${IMAGENET_DATA_PATH} ${QAT2_IC_OPS_TO_QUANTIZE})
	
	### QATv2 for NLP

	set(NLP_DATA_ARCHIVE "Ernie_dataset.tar.gz")
	set(NLP_DATA_DIR "${INFERENCE_DEMO_INSTALL_DIR}/Ernie_dataset")
	set(NLP_DATA_PATH "${NLP_DATA_DIR}/Ernie_dataset/1.8w.bs1")
	set(NLP_LABLES_PATH "${NLP_DATA_DIR}/Ernie_dataset/label.xnli.dev")
	download_qat_data(${NLP_DATA_DIR} ${NLP_DATA_ARCHIVE})

	# QAT2 Ernie
	set(QAT2_ERNIE_MODEL_ARCHIVE "ernie_qat.tar.gz")
	set(QAT2_ERNIE_MODEL_DIR "${QAT_INSTALL_DIR}/Ernie_qat")
	download_qat_model(${QAT2_ERNIE_MODEL_DIR} ${QAT2_ERNIE_MODEL_ARCHIVE})
	set(FP32_ERNIE_MODEL_ARCHIVE "ernie_fp32_model.tar.gz")
	set(FP32_ERNIE_MODEL_DIR "${QAT_INSTALL_DIR}/Ernie_float")
	download_qat_fp32_model(${FP32_ERNIE_MODEL_DIR} ${FP32_ERNIE_MODEL_ARCHIVE})
	inference_qat2_int8_nlp_test(test_qat2_int8_ernie_mkldnn ${QAT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${FP32_ERNIE_MODEL_DIR}/ernie_fp32_model ${NLP_DATA_PATH} ${NLP_LABLES_PATH})

	### Save QAT2 FP32 model or QAT2 INT8 model
        
	set(QAT2_INT8_RESNET50_SAVE_PATH "${QAT_INSTALL_DIR}/ResNet50_qat2_int8")
	set(QAT2_FP32_RESNET50_SAVE_PATH "${QAT_INSTALL_DIR}/ResNet50_qat2_fp32")
	save_qat_ic_model_test(save_qat2_model_resnet50 ${QAT2_RESNET50_MODEL_DIR}/ResNet50_qat_perf/float ${QAT2_FP32_RESNET50_SAVE_PATH} ${QAT2_INT8_RESNET50_SAVE_PATH} ${QAT2_IC_OPS_TO_QUANTIZE})

	set(QAT2_INT8_ERNIE_SAVE_PATH "${QAT_INSTALL_DIR}/Ernie_qat2_int8")
	set(QAT2_FP32_ERNIE_SAVE_PATH "${QAT_INSTALL_DIR}/Ernie_qat2_fp32")
	save_qat_nlp_model_test(save_qat2_model_ernie ${QAT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${QAT2_FP32_ERNIE_SAVE_PATH} ${QAT2_INT8_ERNIE_SAVE_PATH})

	# Convert QAT2 model to dot and pdf files 
	set(QAT2_INT8_ERNIE_DOT_SAVE_PATH "${QAT_INSTALL_DIR}/Ernie_qat2_int8_dot_file")
	convert_model2dot_test(convert_model2dot_ernie ${QAT2_ERNIE_MODEL_DIR}/Ernie_qat/float ${QAT2_INT8_ERNIE_DOT_SAVE_PATH} "Ernie_qat2_int8")

endif()

# Since the tests for QAT FP32 & INT8 comparison support only testing on Linux 
# with MKL-DNN, we remove it here to not test it on other systems.
list(REMOVE_ITEM TEST_OPS
	test_mkldnn_int8_quantization_strategy
	qat_int8_image_classification_comparison
	qat_int8_nlp_comparison)

foreach(src ${TEST_OPS})
    py_test(${src} SRCS ${src}.py)
endforeach()
