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


set(inference_deps paddle_inference_api paddle_fluid_api analysis pass ir_pass_manager naive_executor ${GLOB_PASS_LIB}
        )

if(WITH_GPU AND TENSORRT_FOUND)
    set(inference_deps ${inference_deps} paddle_inference_tensorrt_subgraph_engine analysis_predictor)
endif()

function(inference_api_test TARGET_NAME)
    if (WITH_TESTING)
        set(options "")
        set(oneValueArgs SRC)
        set(multiValueArgs ARGS)
        cmake_parse_arguments(inference_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	if (WITH_GPU)
		cc_test(${TARGET_NAME}
			SRCS ${inference_test_SRC}
			DEPS "${inference_deps}"
			ARGS --dirname=${PYTHON_TESTS_DIR}/book/ --fraction_of_gpu_memory_to_use=0.15)
        else()
		cc_test(${TARGET_NAME}
			SRCS ${inference_test_SRC}
			DEPS "${inference_deps}"
			ARGS --dirname=${PYTHON_TESTS_DIR}/book/)
	endif()
        if(inference_test_ARGS)
            set_tests_properties(${TARGET_NAME}
                    PROPERTIES DEPENDS "${inference_test_ARGS}")
        endif()
    endif(WITH_TESTING)
endfunction(inference_api_test)

cc_library(reset_tensor_array SRCS details/reset_tensor_array.cc DEPS lod_tensor scope)
cc_library(paddle_inference_api SRCS api.cc api_impl.cc helper.cc DEPS reset_tensor_array lod_tensor scope)
cc_library(analysis_predictor SRCS analysis_predictor.cc DEPS paddle_inference_api analysis naive_executor zero_copy_tensor)
cc_library(zero_copy_tensor SRCS details/zero_copy_tensor.cc DEPS paddle_inference_api)
cc_library(zero_copy_tensor_dummy SRCS details/zero_copy_tensor_dummy.cc DEPS paddle_inference_api)
cc_test(test_paddle_inference_api
        SRCS api_tester.cc
        DEPS paddle_inference_api)

inference_api_test(test_api_impl SRC api_impl_tester.cc
                    ARGS test_word2vec test_image_classification)
cc_test(test_analysis_predictor SRCS analysis_predictor_tester.cc DEPS analysis_predictor ${inference_deps} paddle_inference_api
        ARGS --dirname=${PYTHON_TESTS_DIR}/book)

if(WITH_GPU AND TENSORRT_FOUND)
cc_library(paddle_inference_tensorrt_subgraph_engine
        SRCS api_tensorrt_subgraph_engine.cc
        DEPS paddle_inference_api analysis tensorrt_engine paddle_inference_api paddle_fluid_api tensorrt_converter zero_copy_tensor_dummy)

inference_api_test(test_api_tensorrt_subgraph_engine SRC api_tensorrt_subgraph_engine_tester.cc ARGS test_word2vec)
endif()

if (WITH_ANAKIN AND WITH_MKL) # only needed in CI
    # compile the libinference_anakin_api.a and anakin.so.
    cc_library(inference_anakin_api SRCS api.cc api_anakin_engine.cc DEPS anakin_shared anakin_saber mklml scope zero_copy_tensor_dummy)
    cc_library(inference_anakin_api_shared SHARED SRCS api.cc api_anakin_engine.cc DEPS anakin_shared anakin_saber scope)
    function(anakin_target target_name)
      target_compile_options(${target_name} BEFORE PUBLIC ${ANAKIN_COMPILE_EXTRA_FLAGS})
    endfunction()
    anakin_target(inference_anakin_api)
    anakin_target(inference_anakin_api_shared)
endif()
