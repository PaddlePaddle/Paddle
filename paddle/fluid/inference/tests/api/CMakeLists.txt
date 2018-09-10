function (inference_download_and_uncompress install_dir url)
    get_filename_component(filename ${url} NAME)
    message(STATUS "Download inference test stuff ${filename} from ${url}")
    execute_process(COMMAND bash -c "mkdir -p ${install_dir}")
    execute_process(COMMAND bash -c "cd ${install_dir} && wget -q ${url}")
    execute_process(COMMAND bash -c "cd ${install_dir} && tar xzf ${filename}")
    message(STATUS "finish downloading ${filename}")
endfunction(inference_download_and_uncompress)

function(download_model_and_data install_dir model_url data_url)
    if (NOT EXISTS ${install_dir} AND WITH_INFERENCE)
        inference_download_and_uncompress(${install_dir} ${model_url})
        inference_download_and_uncompress(${install_dir} ${data_url})
    endif()
endfunction()

# RNN1
set(RNN1_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/rnn1%2Fmodel.tar.gz")
set(RNN1_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/rnn1%2Fdata.txt.tar.gz")
set(RNN1_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/rnn1")
download_model_and_data(${RNN1_INSTALL_DIR} ${RNN1_MODEL_URL} ${RNN1_DATA_URL})
inference_analysis_test(test_analyzer_rnn1 SRCS analyzer_rnn1_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor
    ARGS --infer_model=${RNN1_INSTALL_DIR}/model
         --infer_data=${RNN1_INSTALL_DIR}/data.txt)

# chinese_ner
set(CHINESE_NER_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/chinese_ner_model.tar.gz")
set(CHINESE_NER_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/chinese_ner-data.txt.tar.gz")
set(CHINESE_NER_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/chinese_ner")
download_model_and_data(${CHINESE_NER_INSTALL_DIR} ${CHINESE_NER_MODEL_URL} ${CHINESE_NER_DATA_URL})
inference_analysis_test(test_analyzer_ner SRCS analyzer_ner_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api analysis_predictor
    ARGS --infer_model=${CHINESE_NER_INSTALL_DIR}/model
        --infer_data=${CHINESE_NER_INSTALL_DIR}/data.txt)

# lac
set(LAC_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/lac_model.tar.gz")
set(LAC_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/lac_data.txt.tar.gz")
set(LAC_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/lac")
download_model_and_data(${LAC_INSTALL_DIR} ${LAC_MODEL_URL} ${LAC_DATA_URL})
inference_analysis_test(test_analyzer_lac SRCS analyzer_lac_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor
    ARGS --infer_model=${LAC_INSTALL_DIR}/model
        --infer_data=${LAC_INSTALL_DIR}/data.txt)

# text_classification
set(TEXT_CLASSIFICATION_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/text-classification-Senta.tar.gz")
set(TEXT_CLASSIFICATION_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/text_classification_data.txt.tar.gz")
set(TEXT_CLASSIFICATION_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/text_classification")
download_model_and_data(${TEXT_CLASSIFICATION_INSTALL_DIR} ${TEXT_CLASSIFICATION_MODEL_URL} ${TEXT_CLASSIFICATION_DATA_URL})
inference_analysis_test(test_text_classification SRCS analyzer_text_classification_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api analysis_predictor
    ARGS --infer_model=${TEXT_CLASSIFICATION_INSTALL_DIR}/text-classification-Senta
         --infer_data=${TEXT_CLASSIFICATION_INSTALL_DIR}/data.txt
         --topn=1 # Just run top 1 batch.
         )
