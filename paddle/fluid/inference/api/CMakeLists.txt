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

cc_library(paddle_inference_api SRCS api.cc api_impl.cc DEPS lod_tensor)

cc_test(test_paddle_inference_api
        SRCS api_tester.cc
        DEPS paddle_inference_api)

inference_api_test(test_api_impl SRC api_impl_tester.cc
                    ARGS test_word2vec test_image_classification)

if(WITH_GPU AND TENSORRT_FOUND)
cc_library(paddle_inference_tensorrt_subgraph_engine
        SRCS api_tensorrt_subgraph_engine.cc
        DEPS paddle_inference_api analysis tensorrt_engine paddle_inference_api paddle_fluid_api tensorrt_converter)

inference_api_test(test_api_tensorrt_subgraph_engine SRC api_tensorrt_subgraph_engine_tester.cc ARGS test_word2vec)
endif()

if (WITH_ANAKIN) # only needed in CI
    # compile the libinference_anakin_api.a and anakin.so.
    nv_library(inference_anakin_api SRCS api.cc api_anakin_engine.cc DEPS anakin_shared anakin_saber)
    #nv_library(inference_anakin_api_shared SHARED SRCS api.cc api_anakin_engine.cc DEPS anakin)
    function(anakin_target target_name)
      target_compile_options(${target_name} BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
    endfunction()
    anakin_target(inference_anakin_api)
    #anakin_target(inference_anakin_api_shared)
    if (WITH_TESTING)
        # TODO(luotao): anakin_model will move to demo_ci later
        set(ANAKIN_MODLE_URL "http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2.anakin.bin")
        execute_process(COMMAND bash -c "wget -q --no-check-certificate ${ANAKIN_MODLE_URL}")
        cc_test(inference_anakin_test SRCS api_anakin_engine_tester.cc
                ARGS --model=${CMAKE_CURRENT_BINARY_DIR}/mobilenet_v2.anakin.bin
                DEPS inference_anakin_api dynload_cuda SERIAL)
        target_compile_options(inference_anakin_test BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
    endif(WITH_TESTING)
endif()
