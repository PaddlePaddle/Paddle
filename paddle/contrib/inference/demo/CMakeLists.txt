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

inference_api_test(simple_on_word2vec ARGS test_word2vec)

set(DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo")
set(mobilenet_url http://paddlemodels.bj.bcebos.com/inference-vis-demos%2Fmobilenet.tar.gz)
set(se_renext50_url http://paddlemodels.bj.bcebos.com/inference-vis-demos%2Fse-renext50.tar.gz)

function(inference_download_test_demo TARGET)
    if (NOT WITH_TESTING)
        return()
    endif()
    set(options "")
    set(oneValueArgs URL)
    set(multiValueArgs SRCS)
    cmake_parse_arguments(tests "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(test_dir "${DEMO_INSTALL_DIR}/${TARGET}")
    message(STATUS "inference demo ${test_dir}")

    if(NOT EXISTS "${test_dir}")
        message(STATUS "Download ${TARGET} model from ${tests_URL}")
        execute_process(COMMAND bash -c "mkdir -p ${test_dir}")
        execute_process(COMMAND bash -c "cd ${test_dir}; wget -q ${tests_URL}")
        execute_process(COMMAND bash -c "cd ${test_dir}; tar xzf *.tar.gz")
    endif()

    cc_test(${TARGET} SRCS "${tests_SRCS}"
        DEPS paddle_inference_api paddle_fluid
        ARGS --data=${test_dir}/data.txt
             --modeldir=${test_dir}/model)
endfunction()

inference_download_test_demo(mobilenet_inference_demo
    SRCS mobilenet.cc
    URL ${mobilenet_url})
inference_download_test_demo(se_renext50_inference_demo
    SRCS se_renext50.cc
    URL ${se_renext50_url})
