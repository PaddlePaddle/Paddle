cc_library(ir_pass_manager SRCS ir_pass_manager.cc DEPS graph pass)
set(analysis_deps
    framework_proto proto_desc ir_pass_manager graph pass paddle_fluid_api executor)

cc_library(analysis SRCS pass_manager.cc node.cc data_flow_graph.cc graph_traits.cc subgraph_splitter.cc
  analyzer.cc
  helper.cc
  # passes
  fluid_to_data_flow_graph_pass.cc
  data_flow_graph_to_fluid_pass.cc
  dfg_graphviz_draw_pass.cc
  tensorrt_subgraph_pass.cc
  tensorrt_subgraph_node_mark_pass.cc
  fluid_to_ir_pass.cc
  model_store_pass.cc
  DEPS ${analysis_deps})

cc_test(test_node SRCS node_tester.cc DEPS analysis)
cc_test(test_dot SRCS dot_tester.cc DEPS analysis)
cc_binary(inference_analyzer SRCS analyzer_main.cc DEPS analysis paddle_fluid)

set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)

function (inference_analysis_test TARGET)
    if(WITH_TESTING)
        set(options "")
        set(oneValueArgs "")
        set(multiValueArgs SRCS ARGS EXTRA_DEPS)
        cmake_parse_arguments(analysis_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        set(mem_opt "")
        if(WITH_GPU)
            set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
        endif()
        cc_test(${TARGET}
                SRCS "${analysis_test_SRCS}"
                DEPS analysis pass ${GLOB_PASS_LIB} ${analysis_test_EXTRA_DEPS}
                ARGS --inference_model_dir=${PYTHON_TESTS_DIR}/book/word2vec.inference.model ${mem_opt} ${analysis_test_ARGS})
        set_tests_properties(${TARGET} PROPERTIES DEPENDS test_word2vec)
    endif(WITH_TESTING)
endfunction(inference_analysis_test)

function (inference_download_and_uncompress install_dir url gz_filename)
    message(STATUS "Download inference test stuff ${gz_filename} from ${url}")
    execute_process(COMMAND bash -c "mkdir -p ${install_dir}")
    execute_process(COMMAND bash -c "cd ${install_dir} && wget -q ${url}")
    execute_process(COMMAND bash -c "cd ${install_dir} && tar xzf ${gz_filename}")
    message(STATUS "finish downloading ${gz_filename}")
endfunction(inference_download_and_uncompress)

set(DITU_RNN_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/ditu_rnn_fluid%2Fmodel.tar.gz")
set(DITU_RNN_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/ditu_rnn_fluid%2Fdata.txt.tar.gz")
set(DITU_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/ditu_rnn" CACHE PATH "Ditu RNN model and data root." FORCE)
if (NOT EXISTS ${DITU_INSTALL_DIR} AND WITH_TESTING)
  inference_download_and_uncompress(${DITU_INSTALL_DIR} ${DITU_RNN_MODEL_URL} "ditu_rnn_fluid%2Fmodel.tar.gz")
  inference_download_and_uncompress(${DITU_INSTALL_DIR} ${DITU_RNN_DATA_URL} "ditu_rnn_fluid%2Fdata.txt.tar.gz")
endif()

inference_analysis_test(test_analyzer SRCS analyzer_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor
    ARGS --infer_ditu_rnn_model=${DITU_INSTALL_DIR}/model
        --infer_ditu_rnn_data=${DITU_INSTALL_DIR}/data.txt)

inference_analysis_test(test_data_flow_graph SRCS data_flow_graph_tester.cc)
inference_analysis_test(test_data_flow_graph_to_fluid_pass SRCS data_flow_graph_to_fluid_pass_tester.cc)
inference_analysis_test(test_fluid_to_ir_pass SRCS fluid_to_ir_pass_tester.cc)
inference_analysis_test(test_fluid_to_data_flow_graph_pass SRCS fluid_to_data_flow_graph_pass_tester.cc)
inference_analysis_test(test_subgraph_splitter SRCS subgraph_splitter_tester.cc)
inference_analysis_test(test_dfg_graphviz_draw_pass SRCS dfg_graphviz_draw_pass_tester.cc)
inference_analysis_test(test_tensorrt_subgraph_pass SRCS tensorrt_subgraph_pass_tester.cc)
inference_analysis_test(test_pass_manager SRCS pass_manager_tester.cc)
inference_analysis_test(test_tensorrt_subgraph_node_mark_pass SRCS tensorrt_subgraph_node_mark_pass_tester.cc)
inference_analysis_test(test_model_store_pass SRCS model_store_pass_tester.cc)

set(CHINESE_NER_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/chinese_ner_model.tar.gz")
set(CHINESE_NER_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/chinese_ner-data.txt.tar.gz")
set(CHINESE_NER_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/chinese_ner" CACHE PATH "Chinese ner model and data root." FORCE)
if (NOT EXISTS ${CHINESE_NER_INSTALL_DIR} AND WITH_TESTING)
  inference_download_and_uncompress(${CHINESE_NER_INSTALL_DIR} ${CHINESE_NER_MODEL_URL} "chinese_ner_model.tar.gz")
  inference_download_and_uncompress(${CHINESE_NER_INSTALL_DIR} ${CHINESE_NER_DATA_URL} "chinese_ner-data.txt.tar.gz")
endif()

inference_analysis_test(test_analyzer_ner SRCS analyzer_ner_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api
    ARGS --infer_model=${CHINESE_NER_INSTALL_DIR}/model
        --infer_data=${CHINESE_NER_INSTALL_DIR}/data.txt)

set(LAC_MODEL_URL "http://paddle-inference-dist.bj.bcebos.com/lac_model.tar.gz")
set(LAC_DATA_URL "http://paddle-inference-dist.bj.bcebos.com/lac_data.txt.tar.gz")
set(LAC_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo/lac" CACHE PATH "LAC model and data root." FORCE)
if (NOT EXISTS ${LAC_INSTALL_DIR} AND WITH_TESTING)
    inference_download_and_uncompress(${LAC_INSTALL_DIR} ${LAC_MODEL_URL} "lac_model.tar.gz")
    inference_download_and_uncompress(${LAC_INSTALL_DIR} ${LAC_DATA_URL} "lac_data.txt.tar.gz")
endif()

inference_analysis_test(test_analyzer_lac SRCS analyzer_lac_tester.cc
    EXTRA_DEPS paddle_inference_api paddle_fluid_api
    ARGS --infer_model=${LAC_INSTALL_DIR}/model
        --infer_data=${LAC_INSTALL_DIR}/data.txt)
