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

function(inference_api_test TARGET_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs ARGS)
    cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
    cc_test(test_paddle_inference_${TARGET_NAME}
            SRCS test_paddle_inference_${TARGET_NAME}.cc
            DEPS paddle_fluid_api paddle_inference_api
            ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
    if(inference_test_ARGS)
        set_tests_properties(test_paddle_inference_${TARGET_NAME}
                 PROPERTIES DEPENDS "${inference_test_ARGS}")
    endif()
endfunction(inference_api_test)


cc_library(paddle_inference_api
    SRCS paddle_inference_api.cc paddle_inference_api_impl.cc
    DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

if(WITH_TESTING)
    cc_test(test_paddle_inference_api
            SRCS test_paddle_inference_api.cc
            DEPS paddle_inference_api)


    add_subdirectory(demo)
    inference_api_test(test_paddle_inference_api_impl
                       ARGS test_word2vec test_image_classification)
endif()
