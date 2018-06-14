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


set(inference_deps paddle_inference_api paddle_fluid_api)

function(inference_api_test TARGET_NAME)
    if (WITH_TESTING)
        set(options "")
        set(oneValueArgs "")
        set(multiValueArgs ARGS)
        cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

        set(PYTHON_TESTS_DIR ${PADDLE_BINARY_DIR}/python/paddle/fluid/tests)
        cc_test(${TARGET_NAME}
                SRCS ${TARGET_NAME}.cc
                DEPS "${inference_deps}"
                ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
        if(inference_test_ARGS)
            set_tests_properties(${TARGET_NAME}
                    PROPERTIES DEPENDS "${inference_test_ARGS}")
        endif()
    endif(WITH_TESTING)
endfunction(inference_api_test)

cc_library(paddle_inference_api
    SRCS paddle_inference_api.cc paddle_inference_api_impl.cc
    DEPS ${FLUID_CORE_MODULES} ${GLOB_OP_LIB})

cc_test(test_paddle_inference_api
        SRCS test_paddle_inference_api.cc
        DEPS paddle_inference_api)

inference_api_test(test_paddle_inference_api_impl
                    ARGS test_word2vec test_image_classification)

if (WITH_ANAKIN AND WITH_TESTING) # only needed in CI
    # Due to Anakin do not have official library releases and the versions of protobuf and cuda do not match Paddle's,
    # so anakin library will not be merged to our official inference library. To use anakin prediction API, one need to
    # compile the libinference_anakin_api.a and compile with anakin.so.
    nv_library(inference_anakin_api SHARED SRCS paddle_inference_api.cc paddle_inference_api_anakin_engine.cc)
    target_compile_options(inference_anakin_api BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
    target_link_libraries(inference_anakin_api anakin anakin_saber_common)
    cc_test(inference_anakin_test SRCS paddle_inference_api_anakin_engine_tester.cc
                                  ARGS --model=${ANAKIN_INSTALL_DIR}/mobilenet_v2.anakin.bin
                                  DEPS inference_anakin_api)
    target_compile_options(inference_anakin_test BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
endif()

if(WITH_TESTING)
    add_subdirectory(demo)
endif()
