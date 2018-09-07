set(INFERENCE_URL "http://paddle-inference-dist.bj.bcebos.com")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo")
set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor)
function (inference_download_and_uncompress install_dir filename)
    message(STATUS "Download inference test stuff from ${INFERENCE_URL}/${filename}")
    execute_process(COMMAND bash -c "mkdir -p ${install_dir}")
    execute_process(COMMAND bash -c "cd ${install_dir} && wget -q ${INFERENCE_URL}/${filename}")
    execute_process(COMMAND bash -c "cd ${install_dir} && tar xzf ${filename}")
    message(STATUS "finish downloading ${filename}")
endfunction(inference_download_and_uncompress)

function(download_model_and_data install_dir model_name data_name)
    if (NOT EXISTS ${install_dir} AND WITH_INFERENCE)
        inference_download_and_uncompress(${install_dir} ${model_name})
        inference_download_and_uncompress(${install_dir} ${data_name})
    endif()
endfunction()

# RNN1
set(RNN1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn1")
download_model_and_data(${RNN1_INSTALL_DIR} "rnn1%2Fmodel.tar.gz" "rnn1%2Fdata.txt.tar.gz")
inference_analysis_test(test_analyzer_rnn1 SRCS analyzer_rnn1_tester.cc 
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${RNN1_INSTALL_DIR}/model
         --infer_data=${RNN1_INSTALL_DIR}/data.txt)

# RNN2
set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
download_model_and_data(${RNN2_INSTALL_DIR} "rnn2_model.tar.gz" "rnn2_data.txt.tar.gz")
inference_analysis_test(test_analyzer_rnn2 SRCS analyzer_rnn2_tester.cc
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${RNN2_INSTALL_DIR}/model
         --infer_data=${RNN2_INSTALL_DIR}/data.txt)

# chinese_ner
set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
download_model_and_data(${CHINESE_NER_INSTALL_DIR} "chinese_ner_model.tar.gz" "chinese_ner-data.txt.tar.gz")
inference_analysis_test(test_analyzer_ner SRCS analyzer_ner_tester.cc
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${CHINESE_NER_INSTALL_DIR}/model
        --infer_data=${CHINESE_NER_INSTALL_DIR}/data.txt)

# lac
set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
download_model_and_data(${LAC_INSTALL_DIR} "lac_model.tar.gz" "lac_data.txt.tar.gz")
inference_analysis_test(test_analyzer_lac SRCS analyzer_lac_tester.cc
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${LAC_INSTALL_DIR}/model
        --infer_data=${LAC_INSTALL_DIR}/data.txt)

# text_classification
set(TEXT_CLASSIFICATION_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
download_model_and_data(${TEXT_CLASSIFICATION_INSTALL_DIR} "text-classification-Senta.tar.gz" "text_classification_data.txt.tar.gz")
inference_analysis_test(test_analyzer_text_classification SRCS analyzer_text_classification_tester.cc
    EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
    ARGS --infer_model=${TEXT_CLASSIFICATION_INSTALL_DIR}/text-classification-Senta
         --infer_data=${TEXT_CLASSIFICATION_INSTALL_DIR}/data.txt)

# ResNet50
set(RESNET50_MODEL "${THIRD_PARTY_PATH}/inference_demo/resnet50/resnet50-model")
inference_analysis_test(test_resnet50 SRCS analyzer_resnet50_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api mkldnn_analysis_predictor
    ARGS --infer_model=${RESNET50_MODEL})
