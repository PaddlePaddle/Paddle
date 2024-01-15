if(NOT WITH_GPU OR NOT WITH_TENSORRT_LLM)
  message(STATUS "NOT WITH_GPU OR NOT WITH_TENSORRT_LLM. ")
  return()
else()
  message(STATUS "WITH_GPU AND WITH_TENSORRT_LLM. ")
endif()

if(WIN32)
  return()
else()
  set(TENSORRT_LLM_ROOT
      "/usr"
      CACHE PATH "TENSORRT LLM ROOT")
  set(TRLLM_INFER_RT libtensorrt_llm.so)
  set(TRLLM_INFER_PLUGIN_RT libnvinfer_plugin_tensorrt_llm.so)
endif()

set(TENSORRT_LLM_INCLUDE_DIR
    "${TENSORRT_LLM_ROOT}/cpp/tensorrt_llm/plugins/api/")

find_path(
  TENSORRTLLM_LIBRARY_DIR
  NAMES ${TRLLM_INFER_RT}
  PATHS ${TENSORRT_LLM_ROOT}/cpp/build/tensorrt_llm
  NO_DEFAULT_PATH
  DOC "Path to TensorRT_LLM library.")

find_path(
  TENSORRTLLM_PLUGIN_LIBRARY_DIR
  NAMES ${TRLLM_INFER_PLUGIN_RT}
  PATHS ${TENSORRT_LLM_ROOT}/cpp/build/tensorrt_llm/plugins
  NO_DEFAULT_PATH
  DOC "Path to TensorRT_LLM'plugins library.")

find_library(
  TENSORRTLLM_LIBRARY
  NAMES ${TRLLM_INFER_RT}
  PATHS ${TENSORRTLLM_LIBRARY_DIR}
  NO_DEFAULT_PATH
  DOC "Path to TensorRTLLM library.")

find_library(
  TENSORRTLLM_PLUGIN_LIBRARY
  NAMES ${TRLLM_INFER_PLUGIN_RT}
  PATHS ${TENSORRTLLM_PLUGIN_LIBRARY_DIR}
  NO_DEFAULT_PATH
  DOC "Path to TensorRTLLM Plugin library.")

if(TENSORRTLLM_LIBRARY AND TENSORRTLLM_PLUGIN_LIBRARY)
  message(STATUS "TensorRT_LLM PATH:" ${TENSORRT_LLM_ROOT})
  set(TENSORRT_LLM_FOUND ON)
  include_directories(${TENSORRT_LLM_INCLUDE_DIR})
  link_directories(${TENSORRTLLM_LIBRARY})
  link_directories(${TENSORRTLLM_PLUGIN_LIBRARY})
  add_definitions(-DPADDLE_WITH_TENSORRT_LLM)
else()
  set(TENSORRT_LLM_FOUND OFF)
  message(
    WARNING
      "TensorRT_LLM is disabled. You are compiling PaddlePaddle with option -DWITH_TENSORRT_LLM=ON, but TensorRT_LLM is not found, please configure path to TensorRT_LLM with option -DTENSORRT_LLM_ROOT or install it."
  )
endif()
