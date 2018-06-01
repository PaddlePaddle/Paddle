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

function(inference_api_demo TARGET_NAME TEST_SRC)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs ARGS)
    cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
    set(arg_list "")
    if(inference_test_ARGS)
        foreach(arg ${inference_test_ARGS})
            list(APPEND arg_list "_${arg}")
        endforeach()
    else()
        list(APPEND arg_list "_")
    endif()
    foreach(arg ${arg_list})
        string(REGEX REPLACE "^_$" "" arg "${arg}")
        cc_test(${TARGET_NAME}
                SRCS ${TEST_SRC}
                DEPS paddle_fluid_api
                ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
        # TODO(panyx0178): Figure out how to add word2vec and image_classification
        # as deps.
        # set_tests_properties(${TARGET_NAME}
        #         PROPERTIES DEPENDS ${DEP_TEST})
    endforeach()
endfunction(inference_api_demo)

inference_api_demo(inference_word2vec_demo simple_on_word2vec.cc)