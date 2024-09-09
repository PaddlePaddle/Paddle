# In Windows, c_api test link must link both 2 shared to avoid symbols redefinition,
# in Linux, c_api test cant do like this or graph_to_program register more than once.
# Both Windows and Linux can only use paddle_inference_c, but this will increase size
# of build folder by 30G.
set(inference_api_tester_deps paddle_inference_api analysis_config)

if(WITH_TESTING AND WITH_INFERENCE_API_TEST)
  function(download_data install_dir data_file check_sum)
    string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
    if(NOT EXISTS ${install_dir}/${file_name})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_data_without_verify install_dir data_file)
    string(REGEX MATCH "[^/\\]+$" file_name ${data_file})
    if(NOT EXISTS ${install_dir}/${file_name})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL} ${data_file})
    endif()
  endfunction()

  function(download_int8_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/int8
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_int8_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/int8 ${data_file})
    endif()
  endfunction()

  function(download_bfloat16_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(
        ${install_dir} ${INFERENCE_URL}/bfloat16 ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_bfloat16_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/bfloat16 ${data_file})
    endif()
  endfunction()

  function(download_GRU_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(${install_dir} ${INFERENCE_URL}/gru
                                        ${data_file} ${check_sum})
    endif()
  endfunction()

  function(download_GRU_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/gru ${data_file})
    endif()
  endfunction()

  function(download_quant_data install_dir data_file check_sum)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress(
        ${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file}
        ${check_sum})
    endif()
  endfunction()

  function(download_quant_data_without_verify install_dir data_file)
    if(NOT EXISTS ${install_dir}/${data_file})
      inference_download_and_uncompress_without_verify(
        ${install_dir} ${INFERENCE_URL}/int8/QAT_models ${data_file})
    endif()
  endfunction()

  function(download_model_and_data install_dir model_name model_check_sum
           data_name data_check_sum)
    download_data(${install_dir} ${model_name} ${model_check_sum})
    download_data(${install_dir} ${data_name} ${data_check_sum})
  endfunction()

  function(download_model_and_data_without_verify install_dir model_name
           data_name)
    download_data_without_verify(${install_dir} ${model_name})
    download_data_without_verify(${install_dir} ${data_name})
  endfunction()

  function(download_result install_dir result_name check_sum)
    download_data(${install_dir} ${result_name} ${check_sum})
  endfunction()

  function(download_result_without_verify install_dir result_name)
    download_data_without_verify(${install_dir} ${result_name})
  endfunction()

  function(inference_analysis_api_test target install_dir filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/model
      --infer_data=${install_dir}/data.txt
      --refer_result=${install_dir}/result.txt)
  endfunction()

  function(inference_analysis_api_int8_test target install_dir filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/model
      --infer_data=${install_dir}/data.txt
      --refer_result=${install_dir}/result.txt
      --accuracy=0.8
      --batch_size=5
      --enable_int8_ptq=true)
  endfunction()

  function(inference_multiple_models_analysis_api_test target install_dir
           filename)
    inference_analysis_test(
      ${target}
      SRCS
      ${filename}
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${install_dir}/mobilenet_v2_models/1
      --infer_model2=${install_dir}/mobilenet_v2_models/xx
      --infer_model3=${install_dir}/mobilenet_v2_models/3)
  endfunction()

  function(inference_analysis_api_test_build TARGET_NAME filename)
    inference_analysis_test_build(${TARGET_NAME} SRCS ${filename} EXTRA_DEPS
                                  common paddle_inference_shared)
  endfunction()

  function(inference_analysis_api_int8_test_run TARGET_NAME test_binary
           model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --warmup_batch_size=${WARMUP_BATCH_SIZE}
      --batch_size=50
      --enable_int8_ptq=true
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=2)
  endfunction()

  function(inference_analysis_api_int8_test_run_custom_warmup_batch_size
           TARGET_NAME test_binary model_dir data_path warmup_batch_size)
    set(WARMUP_BATCH_SIZE ${warmup_batch_size})
    inference_analysis_api_int8_test_run(${TARGET_NAME} ${test_binary}
                                         ${model_dir} ${data_path})
  endfunction()

  function(inference_analysis_api_bfloat16_test_run TARGET_NAME test_binary
           model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --batch_size=50
      --enable_bf16=true
      --paddle_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=2)
  endfunction()

  function(inference_analysis_api_object_dection_int8_test_run TARGET_NAME
           test_binary model_dir data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${model_dir}/model
      --infer_data=${data_path}
      --warmup_batch_size=10
      --batch_size=300
      --enable_int8_ptq=true
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --iterations=1)
  endfunction()

  function(inference_analysis_api_test_with_fake_data_build TARGET_NAME
           filename)
    inference_analysis_test_build(${TARGET_NAME} SRCS ${filename} EXTRA_DEPS
                                  common paddle_inference_shared)
  endfunction()

  function(inference_analysis_api_test_with_fake_data_run TARGET_NAME
           test_binary model_dir disable_fc)
    inference_analysis_test_run(
      ${TARGET_NAME} COMMAND ${test_binary} ARGS
      --infer_model=${model_dir}/model --disable_mkldnn_fc=${disable_fc})
  endfunction()

  function(
    inference_analysis_api_quant_test_run
    TARGET_NAME
    test_binary
    fp32_model_dir
    int8_model_dir
    data_path
    enable_int8_qat)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --fp32_model=${fp32_model_dir}
      --int8_model=${int8_model_dir}
      --infer_data=${data_path}
      --batch_size=50
      --enable_int8_qat=${enable_int8_qat}
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=false
      --iterations=2)
  endfunction()

  function(inference_analysis_api_lexical_test_run TARGET_NAME test_binary
           infer_model data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=50
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --iterations=2)
  endfunction()

  function(inference_analysis_api_lexical_bfloat16_test_run TARGET_NAME
           test_binary infer_model data_path)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=50
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --enable_bf16=true
      --iterations=2)
  endfunction()

  function(
    inference_analysis_api_lexical_int8_test_run
    TARGET_NAME
    test_binary
    infer_model
    data_path
    enable_int8_ptq
    enable_int8_qat
    fuse_multi_gru)
    inference_analysis_test_run(
      ${TARGET_NAME}
      COMMAND
      ${test_binary}
      ARGS
      --infer_model=${infer_model}
      --infer_data=${data_path}
      --batch_size=100
      --cpu_num_threads=${CPU_NUM_THREADS_ON_CI}
      --with_accuracy_layer=true
      --use_analysis=true
      --enable_int8_ptq=${enable_int8_ptq}
      --enable_int8_qat=${enable_int8_qat}
      --quantized_accuracy=0.015
      --fuse_multi_gru=${fuse_multi_gru}
      --iterations=4)
  endfunction()

  function(preprocess_data2bin_test_run target py_script_source data_dir
           output_file)
    py_test(${target}
            SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${py_script_source} ARGS
                 --data_dir=${data_dir} --output_file=${output_file} --local)
  endfunction()

  # RNN2
  set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
  download_model_and_data_without_verify(
    ${RNN2_INSTALL_DIR} "rnn2_model.tar.gz" "rnn2_data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_rnn2 ${RNN2_INSTALL_DIR}
                              analyzer_rnn2_tester.cc EXTRA_DEPS common)

  if(WITH_ONNXRUNTIME AND WIN32)
    # Copy onnxruntime for some c++ test in Windows, since the test will
    # be build only in CI, so suppose the generator in Windows is Ninja.
    copy_onnx(test_analyzer_rnn2)
  endif()

  # small DAM
  set(DAM_SMALL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/small_dam")
  download_model_and_data_without_verify(
    ${DAM_SMALL_INSTALL_DIR} "dam_small_model.tar.gz"
    "dam_small_data.txt.tar.gz")
  inference_analysis_test(
    test_analyzer_small_dam
    SRCS
    analyzer_dam_tester.cc
    EXTRA_DEPS
    paddle_inference_shared
    common
    ARGS
    --infer_model=${DAM_SMALL_INSTALL_DIR}/model
    --infer_data=${DAM_SMALL_INSTALL_DIR}/data.txt)

  # chinese_ner
  set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
  download_model_and_data_without_verify(
    ${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz"
    "chinese_ner-data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_ner ${CHINESE_NER_INSTALL_DIR}
                              analyzer_ner_tester.cc EXTRA_DEPS common)

  # lac
  set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
  download_model_and_data(
    ${LAC_INSTALL_DIR} "lac_model.tar.gz" 419ca6eb85f57a01bfe173591910aec5
    "lac_data.txt.tar.gz" 9983539cd6b34fbdc411e43422776bfd)
  inference_analysis_api_test(test_analyzer_lac ${LAC_INSTALL_DIR}
                              analyzer_lac_tester.cc EXTRA_DEPS common)

  # text_classification
  set(TEXT_CLASSIFICATION_INSTALL_DIR
      "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
  download_model_and_data(
    ${TEXT_CLASSIFICATION_INSTALL_DIR} "text-classification-Senta.tar.gz"
    3f0f440313ca50e26184e65ffd5809ab "text_classification_data.txt.tar.gz"
    36ae620020cc3377f45ed330dd36238f)
  inference_analysis_api_test(
    test_analyzer_text_classification ${TEXT_CLASSIFICATION_INSTALL_DIR}
    analyzer_text_classification_tester.cc EXTRA_DEPS common)

  # seq_conv1
  set(SEQ_CONV1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_conv1")
  download_model_and_data_without_verify(
    ${SEQ_CONV1_INSTALL_DIR} "seq_conv1_model.tar.gz"
    "seq_conv1_data.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_seq_conv1 ${SEQ_CONV1_INSTALL_DIR}
                              analyzer_seq_conv1_tester.cc EXTRA_DEPS common)

  # transformer, the dataset only works on batch_size=8 now
  set(TRANSFORMER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/transformer")
  download_model_and_data_without_verify(
    ${TRANSFORMER_INSTALL_DIR} "temp/transformer_model.tar.gz"
    "temp/transformer_data.txt.tar.gz")
  inference_analysis_test(
    test_analyzer_transformer
    SRCS
    analyzer_transformer_compare_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${TRANSFORMER_INSTALL_DIR}/model
    --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt
    --batch_size=8
    --cpu_num_threads=${CPU_NUM_THREADS_ON_CI})
  inference_analysis_test(
    test_analyzer_transformer_profile
    SRCS
    analyzer_transformer_profile_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${TRANSFORMER_INSTALL_DIR}/model
    --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt
    --batch_size=8
    --cpu_num_threads=${CPU_NUM_THREADS_ON_CI})

  # VIT-OCR
  set(VIT_OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/vit")
  if(NOT EXISTS ${VIT_OCR_INSTALL_DIR}/vit_ocr.tgz)
    inference_download_and_uncompress_without_verify(
      ${VIT_OCR_INSTALL_DIR} ${INFERENCE_URL} "ocr/vit_ocr.tgz")
  endif()
  inference_analysis_test(
    test_analyzer_vit_ocr
    SRCS
    analyzer_vit_ocr_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${VIT_OCR_INSTALL_DIR}/vit_ocr/model
    --infer_data=${VIT_OCR_INSTALL_DIR}/vit_ocr/datavit.txt)

  # ocr
  set(OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/ocr")
  if(NOT EXISTS ${OCR_INSTALL_DIR}/ocr.tar.gz)
    inference_download_and_uncompress_without_verify(
      ${OCR_INSTALL_DIR} "http://paddlemodels.bj.bcebos.com/"
      "inference-vis-demos/ocr.tar.gz")
  endif()
  inference_analysis_api_test(test_analyzer_ocr ${OCR_INSTALL_DIR}
                              analyzer_vis_tester.cc EXTRA_DEPS common)

  # mobilenet with transpose op
  set(MOBILENET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet")
  if(NOT EXISTS ${MOBILENET_INSTALL_DIR}/mobilenet.tar.gz)
    inference_download_and_uncompress_without_verify(
      ${MOBILENET_INSTALL_DIR} "http://paddlemodels.bj.bcebos.com/"
      "inference-vis-demos/mobilenet.tar.gz")
  endif()
  inference_analysis_api_test(
    test_analyzer_mobilenet_transpose ${MOBILENET_INSTALL_DIR}
    analyzer_vis_tester.cc EXTRA_DEPS common)

  ### Image classification tests with fake data
  set(IMG_CLASS_TEST_APP "test_analyzer_image_classification")
  set(IMG_CLASS_TEST_APP_SRC "analyzer_image_classification_tester.cc")

  # build test binary to be used in subsequent tests
  inference_analysis_api_test_with_fake_data_build(${IMG_CLASS_TEST_APP}
                                                   ${IMG_CLASS_TEST_APP_SRC})

  # googlenet
  set(GOOGLENET_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/googlenet")
  download_data_without_verify(${GOOGLENET_MODEL_DIR} "googlenet.tar.gz")
  inference_analysis_api_test_with_fake_data_run(
    test_analyzer_googlenet ${IMG_CLASS_TEST_APP} ${GOOGLENET_MODEL_DIR} false)

  # resnet50
  set(RESNET50_MODEL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/resnet50")
  download_data_without_verify(${RESNET50_MODEL_DIR} "resnet50_model.tar.gz")
  inference_analysis_api_test_with_fake_data_run(
    test_analyzer_resnet50 ${IMG_CLASS_TEST_APP} ${RESNET50_MODEL_DIR} true)
  if(WIN32)
    set_tests_properties(test_analyzer_resnet50 PROPERTIES TIMEOUT 200)
  endif()

  # mobilenet with depthwise_conv op
  set(MOBILENET_MODEL_DIR
      "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet_depthwise_conv")
  download_data_without_verify(${MOBILENET_MODEL_DIR} "mobilenet_model.tar.gz")
  inference_analysis_api_test_with_fake_data_run(
    test_analyzer_mobilenet_depthwise_conv ${IMG_CLASS_TEST_APP}
    ${MOBILENET_MODEL_DIR} false)

  # bert, max_len=20, embedding_dim=128
  set(BERT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/bert_emb128")
  download_model_and_data_without_verify(
    ${BERT_INSTALL_DIR} "bert_emb128_model.tar.gz" "bert_data_len20.txt.tar.gz")
  inference_analysis_api_test(test_analyzer_bert ${BERT_INSTALL_DIR}
                              analyzer_bert_tester.cc EXTRA_DEPS common)

  # multiple models prediction
  set(MMP_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/multi_model_prediction")
  download_data_without_verify(${MMP_INSTALL_DIR}
                               PaddleInference/mobilenet_v2_models.tar.gz)
  inference_multiple_models_analysis_api_test(
    test_analyzer_multi_model_prediction ${MMP_INSTALL_DIR}
    analyzer_mmp_tester.cc EXTRA_DEPS common)

  inference_analysis_test(
    test_analyzer_capi_exp_pd_tensor
    SRCS
    analyzer_capi_exp_pd_tensor_tester.cc
    EXTRA_DEPS
    paddle_inference_c_shared
    ARGS
    --infer_model=${MOBILENET_INSTALL_DIR}/model)

  if(NOT APPLE AND NOT WIN32)
    inference_analysis_test(
      test_analyzer_capi_exp_pd_threads
      SRCS
      analyzer_capi_exp_pd_threads_tester.cc
      EXTRA_DEPS
      paddle_inference_c_shared
      ARGS
      --infer_model=${MOBILENET_INSTALL_DIR}/model)
  endif()

  inference_analysis_test(
    test_analyzer_zerocopytensor_tensor
    SRCS
    analyzer_zerocopy_tensor_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${OCR_INSTALL_DIR}/model)

  if(WITH_DISTRIBUTE AND WITH_PSCORE)
    inference_analysis_test(
      test_analyzer_dist_model
      SRCS
      analyzer_dist_model_tester.cc
      EXTRA_DEPS
      common
      paddle_inference_shared
      ARGS
      --infer_model=${OCR_INSTALL_DIR}/model)
  endif()

  inference_analysis_test(
    test_analyzer_paddletensor_tensor
    SRCS
    analyzer_paddle_tensor_tester.cc
    EXTRA_DEPS
    common
    paddle_inference_shared
    ARGS
    --infer_model=${OCR_INSTALL_DIR}/model
    --infer_data=${OCR_INSTALL_DIR}/data.txt
    --refer_result=${OCR_INSTALL_DIR}/result.txt)

  set_tests_properties(test_analyzer_mobilenet_transpose PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_resnet50 PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_ner PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_googlenet PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_small_dam PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_transformer PROPERTIES TIMEOUT 120)
  set_tests_properties(test_analyzer_mobilenet_depthwise_conv PROPERTIES TIMEOUT
                                                                         120)
  if(WITH_GPU)
    set_tests_properties(test_analyzer_bert PROPERTIES TIMEOUT 120)
  endif()

  if(ON_INFER OR WITH_GPU)
    set_tests_properties(test_analyzer_transformer_profile PROPERTIES TIMEOUT
                                                                      120)
  endif()

  if(WITH_TESTING)
    if(NOT APPLE)
      if(NOT EXISTS ${WORD2VEC_INSTALL_DIR}/word2vec.inference.model.tar.gz)
        inference_download_and_uncompress_without_verify(
          ${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL}
          "word2vec.inference.model.tar.gz")
      endif()
      inference_base_test(
        test_api_impl
        SRCS
        api_impl_tester.cc
        DEPS
        common
        paddle_inference_shared
        ARGS
        --word2vec_dirname=${WORD2VEC_MODEL_DIR}
        --book_dirname=${IMG_CLS_RESNET_INSTALL_DIR})
    endif()
  endif()

  if(NOT APPLE)
    inference_base_test(
      test_analysis_predictor
      SRCS
      analysis_predictor_tester.cc
      DEPS
      paddle_inference_shared
      common
      ARGS
      --dirname=${WORD2VEC_MODEL_DIR})
  endif()

  if(WITH_TESTING AND TEST test_api_impl)
    if(NOT APPLE)
      set_tests_properties(test_api_impl PROPERTIES TIMEOUT 120)
    endif()
  endif()
endif()
