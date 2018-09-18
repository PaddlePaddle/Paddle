set(INFERENCE_URL "http://paddle-inference-dist.cdn.bcebos.com")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
    "A path setting inference demo download directories.")
set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor)
function (inference_download install_dir url filename)
    message(STATUS "Download inference test stuff from ${url}/${filename}")
    execute_process(COMMAND bash -c "mkdir -p ${install_dir}")
    execute_process(COMMAND bash -c "cd ${install_dir} && wget -q ${url}/${filename}")
    message(STATUS "finish downloading ${filename}")
endfunction()

function (inference_download_and_uncompress install_dir url filename)
    inference_download(${install_dir} ${url} ${filename})
    execute_process(COMMAND bash -c "cd ${install_dir} && tar xzf ${filename}")
endfunction()

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

# anakin
if (WITH_ANAKIN AND WITH_MKL) # only needed in CI
   # anakin rnn1
   set(ANAKIN_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/anakin")
   set(ANAKIN_RNN1_INSTALL_DIR "${ANAKIN_INSTALL_DIR}/rnn1")
   inference_download(${ANAKIN_RNN1_INSTALL_DIR} ${INFERENCE_URL} "anakin_test%2Fditu_rnn.anakin2.model.bin")
   inference_download(${ANAKIN_RNN1_INSTALL_DIR} ${INFERENCE_URL} "anakin_test%2Fditu_rnn_data.txt")
   cc_test(test_anakin_rnn1 SRCS anakin_rnn1_tester.cc 
           ARGS --model=${ANAKIN_RNN1_INSTALL_DIR}/anakin_test%2Fditu_rnn.anakin2.model.bin 
                --datapath=${ANAKIN_RNN1_INSTALL_DIR}/anakin_test%2Fditu_rnn_data.txt
           DEPS inference_anakin_api_shared SERIAL)
   # anakin mobilenet
   if(WITH_GPU)
       set(ANAKIN_MOBILENET_INSTALL_DIR "${ANAKIN_INSTALL_DIR}/mobilenet")
       inference_download(${ANAKIN_MOBILENET_INSTALL_DIR} ${INFERENCE_URL} "mobilenet_v2.anakin.bin")
       cc_test(test_anakin_mobilenet SRCS anakin_mobilenet_tester.cc 
               ARGS --model=${ANAKIN_MOBILENET_INSTALL_DIR}/mobilenet_v2.anakin.bin
               DEPS inference_anakin_api_shared dynload_cuda SERIAL)
   endif()
endif()
