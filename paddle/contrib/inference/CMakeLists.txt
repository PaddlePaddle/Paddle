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

set(ANAKIN_INCLUDE "" CACHE STRING "root of Anakin header files")
set(ANAKIN_LIBRARY "" CACHE STRING "path of Anakin library")


set(inference_deps paddle_inference_api paddle_fluid_api)

# if anakin is set enable anakin api implementation
if(ANAKIN_INCLUDE_DIR AND ANAKIN_LIBRARY)
    set(ANAKIN_FOUND ON)
else()
    set(ANAKIN_FOUND OFF)
endif()

if (ANAKIN_FOUND)
    # Anakin's code style doesn't follow google c style.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=comment
                                            -Wno-error=reorder
                                            -Wno-error=format
                                            -Wno-error=switch
                                            -Wno-error=return-type
                                            -Wno-error=non-virtual-dtor
                                            -Wno-error=cpp")

    message(STATUS "Anakin for inference is enabled")
    message(STATUS "Anakin is set INCLUDE:${ANAKIN_INCLUDE} LIBRARY:${ANAKIN_LIBRARY}")
    include_directories("${ANAKIN_INCLUDE}")
    # Anakin's source path is a mass, need to set sub-directories trivially.
    include_directories("${ANAKIN_INCLUDE}/saber")
    link_directories("${ANAKIN_LIBRARY}")

    nv_library(inference_anakin_api SRCS paddle_inference_api_anakin_engine.cc)
    target_link_libraries(inference_anakin_api anakin)
    list(APPEND inference_deps inference_anakin_api)
endif()


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

if (ANAKIN_FOUND)
  nv_test(inference_anakin_test SRCS paddle_inference_api_anakin_engine_tester.cc
    DEPS ${inference_deps} protobuf)
endif()

if(WITH_TESTING)
    add_subdirectory(demo)
endif()
