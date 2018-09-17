set(INFERENCE_URL "http://paddle-inference-dist.bj.bcebos.com")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo")
set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor)
function (inference_download_and_uncompress install_dir url filename)
    message(STATUS "Download inference test stuff from ${url}/${filename}")
    execute_process(COMMAND bash -c "mkdir -p ${install_dir}")
    execute_process(COMMAND bash -c "cd ${install_dir} && wget -q ${url}/${filename}")
    execute_process(COMMAND bash -c "cd ${install_dir} && tar xzf ${filename}")
    message(STATUS "finish downloading ${filename}")
endfunction(inference_download_and_uncompress)

function(download_model_and_data install_dir model_name data_name)
    if (NOT EXISTS ${install_dir})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${model_name})
        inference_download_and_uncompress(${install_dir} ${INFERENCE_URL} ${data_name})
    endif()
endfunction()

function(inference_analysis_api_test target install_dir filename)
    inference_analysis_test(${target} SRCS ${filename}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
        ARGS --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt)
endfunction()

# RNN1
if(NOT APPLE)
    set(RNN1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn1")
    download_model_and_data(${RNN1_INSTALL_DIR} "rnn1%2Fmodel.tar.gz" "rnn1%2Fdata.txt.tar.gz")
    inference_analysis_api_test(test_analyzer_rnn1 ${RNN1_INSTALL_DIR} analyzer_rnn1_tester.cc)
else()
    # TODO: fix this test on MACOS, the reason is that
    # fusion_seqexpand_concat_fc_op is not supported on MACOS
    message(WARNING "These tests has been disabled in OSX before being fixed: \n test_analyzer_rnn1")
endif()

# RNN2
set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
download_model_and_data(${RNN2_INSTALL_DIR} "rnn2_model.tar.gz" "rnn2_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_rnn2 ${RNN2_INSTALL_DIR} analyzer_rnn2_tester.cc)

# chinese_ner
set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
download_model_and_data(${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz" "chinese_ner-data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_ner ${CHINESE_NER_INSTALL_DIR} analyzer_ner_tester.cc)

# lac
set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
download_model_and_data(${LAC_INSTALL_DIR} "lac_model.tar.gz" "lac_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_lac ${LAC_INSTALL_DIR} analyzer_lac_tester.cc)

# text_classification
set(TEXT_CLASSIFICATION_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
download_model_and_data(${TEXT_CLASSIFICATION_INSTALL_DIR} "text-classification-Senta.tar.gz" "text_classification_data.txt.tar.gz")
inference_analysis_api_test(test_analyzer_text_classification ${TEXT_CLASSIFICATION_INSTALL_DIR} analyzer_text_classification_tester.cc)

# ocr
set(OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/ocr")
if (NOT EXISTS ${OCR_INSTALL_DIR})
    inference_download_and_uncompress(${OCR_INSTALL_DIR} "http://paddlemodels.cdn.bcebos.com/" "inference-vis-demos%2Focr.tar.gz")
endif()
inference_analysis_api_test(test_analyzer_ocr ${OCR_INSTALL_DIR} analyzer_vis_tester.cc)
