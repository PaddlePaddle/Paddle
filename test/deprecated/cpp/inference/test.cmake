include(ExternalProject)
set(INFERENCE_URL
    "http://paddle-inference-dist.bj.bcebos.com"
    CACHE STRING "inference download url")
set(INFERENCE_DEMO_INSTALL_DIR
    "${THIRD_PARTY_PATH}/inference_demo"
    CACHE STRING "A path setting inference demo download directories.")
set(CPU_NUM_THREADS_ON_CI
    4
    CACHE STRING "Run multi-threads on CI to reduce CI time.")
set(WARMUP_BATCH_SIZE
    100
    CACHE STRING "Default warmup_batch_size.")
function(inference_download INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  ExternalProject_Add(
    extern_inference_download_${FILENAME_EX}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${INSTALL_DIR}
    URL ${URL}/${FILENAME}
    DOWNLOAD_COMMAND wget --no-check-certificate -q -O
                     ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME}
    DOWNLOAD_DIR ${INSTALL_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")
endfunction()

function(inference_download_and_uncompress INSTALL_DIR URL FILENAME CHECK_SUM)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%./\\]" "_" FILENAME_EX ${FILENAME})
  string(REGEX MATCH "[^/\\]+$" DOWNLOAD_NAME ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
    ${EXTERNAL_PROJECT_NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${INSTALL_DIR}
    URL ${URL}/${FILENAME}
    URL_HASH MD5=${CHECK_SUM}
    DOWNLOAD_DIR ${INSTALL_DIR}
    DOWNLOAD_NO_EXTRACT 1
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${INSTALL_DIR} ${CMAKE_COMMAND} -E
                  tar xzf ${DOWNLOAD_NAME}
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")
endfunction()

function(inference_download_and_uncompress_without_verify INSTALL_DIR URL
         FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%./\\]" "_" FILENAME_EX ${FILENAME})
  string(REGEX MATCH "[^/\\]+$" DOWNLOAD_NAME ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  get_property(TARGET_EXIST GLOBAL PROPERTY ${EXTERNAL_PROJECT_NAME})
  if(NOT "${TARGET_EXIST}" STREQUAL EXIST)
    ExternalProject_Add(
      ${EXTERNAL_PROJECT_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX ${INSTALL_DIR}
      URL ${URL}/${FILENAME}
      DOWNLOAD_DIR ${INSTALL_DIR}
      DOWNLOAD_NO_EXTRACT 1
      DOWNLOAD_NO_PROGRESS 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ${CMAKE_COMMAND} -E chdir ${INSTALL_DIR} ${CMAKE_COMMAND} -E
                    tar xzf ${DOWNLOAD_NAME}
      UPDATE_COMMAND ""
      INSTALL_COMMAND "")
    set_property(GLOBAL PROPERTY ${EXTERNAL_PROJECT_NAME} "EXIST")
  endif()
endfunction()

function(inference_base_test_build TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(base_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET} ${base_test_SRCS})
  if(WIN32)
    target_compile_definitions(${TARGET} PUBLIC STATIC_PADDLE)
  endif()
  if("${base_test_DEPS};" MATCHES "paddle_inference_shared;")
    list(REMOVE_ITEM base_test_DEPS paddle_inference_shared)

    target_link_libraries(${TARGET}
                          $<TARGET_LINKER_FILE:paddle_inference_shared>)
    add_dependencies(${TARGET} paddle_inference_shared)

  elseif("${base_test_DEPS};" MATCHES "paddle_inference_c_shared;")
    list(REMOVE_ITEM base_test_DEPS paddle_inference_c_shared)
    target_link_libraries(
      ${TARGET} $<TARGET_LINKER_FILE:paddle_inference_c_shared> common)
    add_dependencies(${TARGET} paddle_inference_c_shared)
  else()
    message(
      FATAL_ERROR
        "inference_base_test_build must link either paddle_inference_shared or paddle_inference_c_shared"
    )
  endif()
  if(NOT ((NOT WITH_PYTHON) AND ON_INFER))
    target_link_libraries(${TARGET} ${PYTHON_LIBRARIES})
  endif()
  if(WITH_SHARED_PHI)
    target_link_libraries(${TARGET} phi)
    add_dependencies(${TARGET} phi)
  endif()
  if(WITH_CINN)
    target_link_libraries(${TARGET} $<TARGET_LINKER_FILE:cinnapi>)
    add_dependencies(${TARGET} cinnapi)
  endif()
  if(WITH_GPU)
    target_link_libraries(${TARGET} ${CUDA_CUDART_LIBRARY})
  endif()
  if(WITH_XPU)
    target_link_libraries(${TARGET} xpulib)
  endif()
  if(WITH_ROCM)
    target_link_libraries(${TARGET} ${ROCM_HIPRTC_LIB})
  endif()
  if(WITH_ONNXRUNTIME)
    target_link_libraries(${TARGET} onnxruntime)
  endif()
  if(APPLE)
    target_link_libraries(
      ${TARGET}
      "-Wl,-rpath,$<TARGET_FILE_DIR:${paddle_lib}> -Wl,-rpath,$<TARGET_FILE_DIR:phi> -Wl,-rpath,$<TARGET_FILE_DIR:pir>"
    )
  endif()
  target_link_libraries(${TARGET} ${base_test_DEPS} paddle_gtest_main_new gtest
                        glog)
  add_dependencies(${TARGET} ${base_test_DEPS} paddle_gtest_main_new)
  common_link(${TARGET})
  check_coverage_opt(${TARGET} ${base_test_SRCS})
endfunction()

function(inference_base_test_run TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs COMMAND ARGS)
  cmake_parse_arguments(base_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(WITH_GPU)
    set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
  endif()
  cc_test_run(${TARGET} COMMAND ${base_test_COMMAND} ARGS ${mem_opt}
              ${base_test_ARGS})
endfunction()

function(inference_base_test TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS ARGS DEPS)
  cmake_parse_arguments(base_test "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  inference_base_test_build(${TARGET} SRCS ${base_test_SRCS} DEPS
                            ${base_test_DEPS})
  inference_base_test_run(${TARGET} COMMAND ${TARGET} ARGS ${base_test_ARGS})
endfunction()

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")
set(WORD2VEC_MODEL_DIR "${WORD2VEC_INSTALL_DIR}/word2vec.inference.model")

if(NOT EXISTS ${WORD2VEC_INSTALL_DIR}/word2vec.inference.model.tar.gz)
  inference_download_and_uncompress_without_verify(
    ${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL} "word2vec.inference.model.tar.gz")
endif()

set(IMG_CLS_RESNET_INSTALL_DIR
    "${INFERENCE_DEMO_INSTALL_DIR}/image_classification_resnet")
set(IMG_CLS_RESNET_MODEL_DIR
    "${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model")

if(NOT EXISTS
   ${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model.tgz
)
  inference_download_and_uncompress_without_verify(
    ${IMG_CLS_RESNET_INSTALL_DIR} ${INFERENCE_URL}
    "image_classification_resnet.inference.model.tgz")
endif()

if(WITH_ONNXRUNTIME)
  set(MOBILENETV2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/MobileNetV2")
  set(MOBILENETV2_MODEL_DIR "${MOBILENETV2_INSTALL_DIR}/MobileNetV2")
endif()
