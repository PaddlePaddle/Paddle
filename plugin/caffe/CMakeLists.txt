# use caffe plugin

set(CAFFE_PATH "/home/dangqingqing/caffe")
if(NOT DEFINED CAFFE_PATH)
  message(FATAL_ERROR "Please set CAFFE_PATH to point to the caffe source installation")
endif()
list(APPEND CMAKE_MODULE_PATH ${CAFFE_PATH}/cmake)
include_directories(${CAFFE_PATH}/include)
include_directories(${CAFFE_PATH}/build/src)
include_directories(${CMAKE_BINARY_DIR}/caffe/include)

set(CAFFE_LINKER_LIBS "")

find_library(CAFFE_LIBRARY NAMES libcaffe.so # libcaffe.a
    PATHS ${CAFFE_PATH}/build/lib
          NO_DEFAULT_PATH
    DOC "Path to caffe library.")
list(APPEND CAFFE_LINKER_LIBS ${CAFFE_LIBRARY})

link_directories(${CAFFE_PATH}/build/lib)

file(GLOB_RECURSE PLUGINS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}" "*.h")
file(GLOB_RECURSE PLUGINS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cpp")
#file(GLOB_RECURSE PLUGINS_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}" "*.cu")

if("${CBLAS_PROVIDER}" STREQUAL "MKL")
    add_definitions(-DUSE_MKL)
endif()

if(CUDNN_FOUND)
    add_definitions(-DUSE_CUDNN)
endif()

if(WITH_GPU)
    #list(APPEND PLUGINS_SOURCES ${PLUGINS_CU_SOURCE})
    cuda_add_library(paddle_plugin_caffe ${PLUGINS_SOURCES})
else()
    add_library(paddle_plugin_caffe STATIC
        ${PLUGINS_SOURCES})
endif()
target_link_libraries(paddle_plugin_caffe ${CAFFE_LINKER_LIBS})
add_dependencies(paddle_plugin_caffe gen_proto_cpp)

#add_style_check_target(paddle_plugin_caffe
#    ${PLUGINS_SOURCES})
#add_style_check_target(paddle_plugin_caffe
#    ${PLUGINS_HEADERS})
#add_style_check_target(paddle_plugin_caffe
#    ${PLUGINS_CU_SOURCES})
