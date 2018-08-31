# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pessimizing-move")
endif(APPLE)


set(inference_deps paddle_inference_api paddle_fluid_api analysis pass ir_pass_manager
  graph_viz_pass fc_fuse_pass
  infer_clean_graph_pass
  )

if(WITH_GPU AND TENSORRT_FOUND)
    set(inference_deps ${inference_deps} paddle_inference_tensorrt_subgraph_engine)
endif()

function(inference_api_test TARGET_NAME)
    if (WITH_TESTING)
        set(options "")
        set(oneValueArgs SRC)
        set(multiValueArgs ARGS)
        cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
        cc_test(${TARGET_NAME}
                SRCS ${inference_test_SRC}
                DEPS "${inference_deps}"
                ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
        if(inference_test_ARGS)
            set_tests_properties(${TARGET_NAME}
                    PROPERTIES DEPENDS "${inference_test_ARGS}")
        endif()
    endif(WITH_TESTING)
endfunction(inference_api_test)

cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS lod_tensor)
cc_library(analysis_predictor SRCS analysis_predictor.cc DEPS paddle_inference_api)

cc_test(test_paddle_inference_api
        SRCS api_tester.cc
        DEPS paddle_inference_api)

inference_api_test(test_api_impl SRC api_impl_tester.cc
                    ARGS test_word2vec test_image_classification)

if(WITH_GPU AND TENSORRT_FOUND)
    set(TRT_INFERENCE_MODEL_DIR ${THIRD_PARTY_PATH}/inference_model/)
    set(TRT_MODEL_FILE "new_giscnn_fluid_model.tar.gz")
    download_inference_file(FILE_NAME ${TRT_MODEL_FILE} COMPRESSED TRUE FILE_SAVE_DIR ${TRT_INFERENCE_MODEL_DIR})

    cc_library(paddle_inference_tensorrt_subgraph_engine
        SRCS api_tensorrt_subgraph_engine.cc
        DEPS paddle_inference_api analysis tensorrt_engine paddle_inference_api paddle_fluid_api tensorrt_converter)

    cc_test(api_tensorrt_engine_cnn_tester SRCS api_tensorrt_engine_cnn_tester.cc 
	ARGS --dirname=${TRT_INFERENCE_MODEL_DIR}/new_giscnn_fluid_model/
	DEPS paddle_inference_tensorrt_subgraph_engine)

    inference_api_test(test_api_tensorrt_subgraph_engine SRC api_tensorrt_subgraph_engine_tester.cc ARGS test_word2vec)
endif()

if (WITH_ANAKIN AND WITH_GPU) # only needed in CI
    # TODO(luotao): ANAKIN_MODLE_*** etc will move to demo ci later.
    # ============ Download model file =============
    set(ANAKIN_MODEL_DIR ${THIRD_PARTY_PATH}/inference_model/anakin/)
    set(ANAKIN_MODEL_FILE "mobilenet_v2.anakin.bin")
    download_inference_file(FILE_NAME ${ANAKIN_MODEL_FILE} COMPRESSED FALSE FILE_SAVE_DIR ${ANAKIN_MODEL_DIR})

    set(ANAKIN_RNN_MODEL_DIR ${THIRD_PARTY_PATH}/inference_model/anakin/rnn/)
    set(ANAKIN_RNN_MODEL_FILE "anakin_test%2Fditu_rnn.anakin2.model.bin")
    set(ANAKIN_RNN_DATA_FILE "anakin_test%2Fditu_rnn_data.txt")
    download_inference_file(FILE_NAME ${ANAKIN_RNN_MODEL_FILE} COMPRESSED FALSE FILE_SAVE_DIR ${ANAKIN_RNN_MODEL_DIR})
    download_inference_file(FILE_NAME ${ANAKIN_RNN_DATA_FILE} COMPRESSED FALSE FILE_SAVE_DIR ${ANAKIN_RNN_MODEL_DIR})

    set(ANAKIN_CNN_MODEL_DIR ${THIRD_PARTY_PATH}/inference_model/anakin/cnn/)
    set(ANAKIN_CNN_MODEL_FILE "new_giscnn.anakin2.bin.tar.gz")
    download_inference_file(FILE_NAME ${ANAKIN_CNN_MODEL_FILE} COMPRESSED TRUE FILE_SAVE_DIR ${ANAKIN_CNN_MODEL_DIR})
    # ============ Download model file =============

    # compile the libinference_anakin_api.a and anakin.so.
    cc_library(inference_anakin_api SRCS api.cc api_anakin_engine.cc DEPS anakin_shared anakin_saber mklml)
    cc_library(inference_anakin_api_shared SHARED SRCS api.cc api_anakin_engine.cc DEPS anakin_shared anakin_saber)
    function(anakin_target target_name)
      target_compile_options(${target_name} BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
    endfunction()
    anakin_target(inference_anakin_api)
    anakin_target(inference_anakin_api_shared)
    if (WITH_TESTING)
        cc_test(api_anakin_engine_cnn_tester SRCS api_anakin_engine_cnn_tester.cc
                ARGS --model=${ANAKIN_CNN_MODEL_DIR}/new_giscnn.anakin2.bin
                DEPS inference_anakin_api_shared dynload_cuda SERIAL)
        cc_test(api_anakin_engine_tester SRCS api_anakin_engine_tester.cc 
                ARGS --model=${ANAKIN_MODEL_DIR}/${ANAKIN_MODEL_FILE}
                DEPS inference_anakin_api_shared dynload_cuda SERIAL)
        cc_test(api_anakin_engine_rnn_tester SRCS api_anakin_engine_rnn_tester.cc 
                ARGS --model=${ANAKIN_RNN_MODEL_DIR}/${ANAKIN_RNN_MODEL_FILE}
                     --datapath=${ANAKIN_RNN_MODEL_DIR}/${ANAKIN_RNN_DATA_FILE}
                DEPS inference_anakin_api_shared dynload_cuda SERIAL)
    endif(WITH_TESTING)
endif()
