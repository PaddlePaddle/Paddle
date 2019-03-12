set(INFERENCE_EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor benchmark)

if(WITH_GPU AND TENSORRT_FOUND)
  set(INFERENCE_EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} analysis ${analysis_deps} ir_pass_manager analysis_predictor)
endif()

function(download_model target_name install_dir model_name)
  if(NOT EXISTS ${install_dir})
    inference_download_and_uncompress(${target_name} ${install_dir} ${INFERENCE_URL} ${model_name})
    set(${target_name} ${${target_name}} PARENT_SCOPE)
  endif()
endfunction()

function(download_model_and_data target_name install_dir model_name data_name)
  if(NOT EXISTS ${install_dir})
    inference_download_and_uncompress(${target_name} ${install_dir} ${INFERENCE_URL} ${model_name})
    inference_download_and_uncompress(${target_name} ${install_dir} ${INFERENCE_URL} ${data_name})
    set(${target_name} ${${target_name}} PARENT_SCOPE)
  endif()
endfunction()

function(inference_analysis_api_test TARGET INSTALL_DIR)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs URL MODEL DATA SRCS ARGS EXTRA_DEPS DOWNLOAD_DEPS)
  cmake_parse_arguments(analysis_api_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(analysis_api_test_URL)
    set(TESTER_URL ${analysis_api_test_URL})
  else()
    set(TESTER_URL ${INFERENCE_URL})
  endif()
  set(download_deps)
  if(analysis_api_test_MODEL)
    inference_download_and_uncompress(download_deps
        ${INSTALL_DIR} ${TESTER_URL} ${analysis_api_test_MODEL})
    if(analysis_api_test_DATA)
      inference_download_and_uncompress(download_deps
          ${INSTALL_DIR} ${TESTER_URL} ${analysis_api_test_DATA})

      set(TESTER_ARGS --infer_model=${INSTALL_DIR}/model --infer_data=${INSTALL_DIR}/data.txt)
    else()
      set(TESTER_ARGS --infer_model=${install_dir}/model)
    endif()
  else()
    message(FATAL_ERROR "Model file should be specified for inference api tester.")
  endif()

  if(analysis_api_test_ARGS)
    set(TESTER_ARGS ${analysis_api_test_ARGS})
  endif()
  if(download_deps)
    inference_analysis_test(${TARGET}
        SRCS ${analysis_api_test_SRCS}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} benchmark ${analysis_api_test_EXTRA_DEPS}
        ARGS ${TESTER_ARGS}
        DOWNLOAD_DEPS ${download_deps})
  else()
    inference_analysis_test(${TARGET}
        SRCS ${analysis_api_test_SRCS}
        EXTRA_DEPS ${INFERENCE_EXTRA_DEPS} benchmark ${analysis_api_test_EXTRA_DEPS}
        ARGS ${TESTER_ARGS})
  endif()
endfunction()


if(NOT APPLE AND WITH_MKLML)
  # RNN1
  set(RNN1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn1")
  inference_analysis_api_test(test_analyzer_rnn1 ${RNN1_INSTALL_DIR}
      MODEL "rnn1%2Fmodel.tar.gz"
      DATA  "rnn1%2Fdata.txt.tar.gz"
      SRCS  analyzer_rnn1_tester.cc SERIAL)
  
  # seq_pool1
  set(SEQ_POOL1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_pool")
  inference_analysis_api_test(test_analyzer_seq_pool1 ${SEQ_POOL1_INSTALL_DIR}
      MODEL "seq_pool1_model_.tar.gz"
      DATA  "seq_pool1_data.txt.tar.gz"
      SRCS  analyzer_seq_pool1_tester.cc SERIAL)
else()
  # TODO: fix this test on MACOS and OPENBLAS, the reason is that
  # fusion_seqexpand_concat_fc_op is not supported on MACOS and OPENBLAS
  message(WARNING "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_rnn1")
  message(WARNING "These tests has been disabled in OSX or WITH_MKL=OFF before being fixed: \n test_analyzer_seq_pool1")
endif()

# RNN2
set(RNN2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/rnn2")
inference_analysis_api_test(test_analyzer_rnn2 ${RNN2_INSTALL_DIR}
    MODEL "rnn2_model.tar.gz"
    DATA  "rnn2_data.txt.tar.gz"
    SRCS  analyzer_rnn2_tester.cc)

# TODO(luotao, Superjom) Disable DAM test, temporarily fix
# https://github.com/PaddlePaddle/Paddle/issues/15032#issuecomment-455990914.
# After inference framework refactor, will reopen it.
# normal DAM
set(DAM_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/dam")
#inference_analysis_api_test(test_analyzer_dam ${DAM_INSTALL_DIR}
#    MODEL "DAM_model.tar.gz"
#    DATA  "DAM_data.txt.tar.gz"
#    SRCS  analyzer_dam_tester.cc
#    EXTRA_DEPS legacy_allocator SERIAL)

# small DAM
set(DAM_SMALL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/small_dam")
inference_analysis_api_test(test_analyzer_small_dam ${DAM_SMALL_INSTALL_DIR}
    MODEL "dam_small_model.tar.gz"
    DATA  "dam_small_data.txt.tar.gz"
    SRCS   analyzer_dam_tester.cc
    ARGS   --infer_model=${DAM_SMALL_INSTALL_DIR}/model --infer_data=${DAM_SMALL_INSTALL_DIR}/data.txt --max_turn_num=1 SERIAL)

# chinese_ner
set(CHINESE_NER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/chinese_ner")
inference_analysis_api_test(test_analyzer_ner ${CHINESE_NER_INSTALL_DIR}
    MODEL "chinese_ner_model.tar.gz"
    DATA  "chinese_ner-data.txt.tar.gz"
    SRCS  analyzer_ner_tester.cc)

# lac
set(LAC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/lac")
inference_analysis_api_test(test_analyzer_lac ${LAC_INSTALL_DIR}
    MODEL "lac_model.tar.gz"
    DATA  "lac_data.txt.tar.gz"
    SRCS  analyzer_lac_tester.cc)

# MM DNN
set(MM_DNN_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mm_dnn")
inference_analysis_api_test(test_analyzer_mm_dnn ${MM_DNN_INSTALL_DIR}
    MODEL "MM_DNN_model.tar.gz"
    DATA  "MM_DNN_data.txt.tar.gz"
    SRCS  analyzer_mm_dnn_tester.cc SERIAL)

# Pyramid DNN
set(PYRAMID_DNN_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/pyramid_dnn")
inference_analysis_api_test(test_analyzer_pyramid_dnn ${PYRAMID_DNN_INSTALL_DIR}
    MODEL "PyramidDNN_model.tar.gz"
    DATA  "PyramidDNN_data.txt.tar.gz"
    SRCS  analyzer_pyramid_dnn_tester.cc)

# text_classification
set(TEXT_CLASSIFICATION_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/text_classification")
inference_analysis_api_test(test_analyzer_text_classification ${TEXT_CLASSIFICATION_INSTALL_DIR}
    MODEL "text-classification-Senta.tar.gz"
    DATA  "text_classification_data.txt.tar.gz"
    SRCS  analyzer_text_classification_tester.cc)

# seq_conv1
set(SEQ_CONV1_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/seq_conv1")
inference_analysis_api_test(test_analyzer_seq_conv1 ${SEQ_CONV1_INSTALL_DIR}
    MODEL "seq_conv1_model.tar.gz"
    DATA  "seq_conv1_data.txt.tar.gz"
    SRCS  analyzer_seq_conv1_tester.cc)

# transformer, the dataset only works on batch_size=8 now
set(TRANSFORMER_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/transformer")
inference_analysis_test(test_analyzer_transformer SRCS analyzer_transformer_tester.cc 
  EXTRA_DEPS ${INFERENCE_EXTRA_DEPS}
  ARGS --infer_model=${TRANSFORMER_INSTALL_DIR}/model --infer_data=${TRANSFORMER_INSTALL_DIR}/data.txt --batch_size=8)

# ocr
set(OCR_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/ocr")
inference_analysis_api_test(test_analyzer_ocr ${OCR_INSTALL_DIR}
    URL   "http://paddlemodels.cdn.bcebos.com/"
    MODEL "inference-vis-demos%2Focr.tar.gz"
    SRCS  analyzer_vis_tester.cc
    ARGS  --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt
          --refer_result=${install_dir}/result.txt
    SERIAL)

# mobilenet with transpose op
set(MOBILENET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet")
inference_analysis_api_test(test_analyzer_mobilenet_transpose ${MOBILENET_INSTALL_DIR}
    URL   "http://paddlemodels.cdn.bcebos.com/"
    MODEL "inference-vis-demos%2Fmobilenet.tar.gz"
    SRCS  analyzer_vis_tester.cc
    ARGS  --infer_model=${install_dir}/model --infer_data=${install_dir}/data.txt
          --refer_result=${install_dir}/result.txt
    SERIAL)

# googlenet
set(GOOGLENET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/googlenet")
inference_analysis_api_test(test_analyzer_googlenet ${GOOGLENET_INSTALL_DIR}
    MODEL "googlenet.tar.gz"
    SRCS  analyzer_resnet50_tester.cc SERIAL)

# resnet50
set(RESNET50_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/resnet50")
inference_analysis_api_test(test_analyzer_resnet50 ${RESNET50_INSTALL_DIR}
    MODEL "resnet50_model.tar.gz"
    SRCS  analyzer_resnet50_tester.cc SERIAL)

# mobilenet with depthwise_conv op
set(MOBILENET_DEPTHWISE_CONV_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/mobilenet_depthwise_conv")
inference_analysis_api_test(test_analyzer_mobilenet_depthwise_conv ${MOBILENET_DEPTHWISE_CONV_INSTALL_DIR}
    MODEL "mobilenet_model.tar.gz"
    SRCS  analyzer_resnet50_tester.cc SERIAL)

# bert, max_len=20, embedding_dim=128
set(BERT_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/bert_emb128")
inference_analysis_api_test(test_analyzer_bert ${BERT_INSTALL_DIR}
    MODEL "bert_emb128_model.tar.gz"
    DATA  "bert_data_len20.txt.tar.gz"
    SRCS  analyzer_bert_tester.cc SERIAL)

# anakin
if (WITH_ANAKIN AND WITH_MKL) # only needed in CI
  # anakin rnn1
  set(ANAKIN_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/anakin")
  set(ANAKIN_RNN1_INSTALL_DIR "${ANAKIN_INSTALL_DIR}/rnn1")
  inference_analysis_api_test(test_anakin_rnn1 ${ANAKIN_RNN1_INSTALL_DIR}
      MODEL "anakin_test%2Fditu_rnn.anakin2.model.bin"
      DATA  "anakin_test%2Fditu_rnn_data.txt"
      SRCS  anakin_rnn1_tester.cc
      ARGS  --model=${ANAKIN_RNN1_INSTALL_DIR}/anakin_test%2Fditu_rnn.anakin2.model.bin
            --datapath=${ANAKIN_RNN1_INSTALL_DIR}/anakin_test%2Fditu_rnn_data.txt
      EXTRA_DEPS inference_anakin_api_shared SERIAL)
  # anakin mobilenet
  if(WITH_GPU)
    set(ANAKIN_MOBILENET_INSTALL_DIR "${ANAKIN_INSTALL_DIR}/mobilenet")
    inference_analysis_api_test(test_anakin_mobilenet ${ANAKIN_MOBILENET_INSTALL_DIR}
        MODEL "mobilenet_v2.anakin.bin"
        SRCS anakin_mobilenet_tester.cc
        ARGS --model=${ANAKIN_MOBILENET_INSTALL_DIR}/mobilenet_v2.anakin.bin
        EXTRA_DEPS inference_anakin_api_shared dynload_cuda SERIAL)
  endif()
endif()

if(WITH_GPU AND TENSORRT_FOUND)
  set(TRT_MODEL_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/trt")
  inference_analysis_api_test(test_trt_models ${INFERENCE_DEMO_INSTALL_DIR}
      MODEL "trt_test_models.tar.gz"
      SRCS  trt_models_tester.cc
      ARGS  --infer_model=${TRT_MODEL_INSTALL_DIR}/trt_test_models SERIAL)
endif()
