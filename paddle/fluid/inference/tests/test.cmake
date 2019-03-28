include(ExternalProject)
set(INFERENCE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
    "A path setting inference demo download directories.")

function(inference_download INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  ExternalProject_Add(
      extern_inference_download_${FILENAME_EX}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${INSTALL_DIR}
      URL                   ${URL}/${FILENAME}
      DOWNLOAD_COMMAND      wget -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME}
      DOWNLOAD_DIR          ${INSTALL_DIR}
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
  )
endfunction()

function(inference_download_and_uncompress INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_inference_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
      ${EXTERNAL_PROJECT_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${INSTALL_DIR}
      DOWNLOAD_COMMAND      wget -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} &&
                            ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
      DOWNLOAD_DIR          ${INSTALL_DIR}
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
  )
endfunction()

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")
if(NOT EXISTS ${WORD2VEC_INSTALL_DIR} AND NOT WIN32)
  inference_download_and_uncompress(${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL} "word2vec.inference.model.tar.gz")
endif()
set(WORD2VEC_MODEL_DIR "${WORD2VEC_INSTALL_DIR}/word2vec.inference.model")

function (inference_base_test TARGET)
   set(options "")
   set(oneValueArgs "")
   set(multiValueArgs SRCS ARGS DEPS)
   cmake_parse_arguments(base_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
   if(WITH_GPU)
       set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
   endif()
   cc_test(${TARGET} SRCS ${base_test_SRCS} DEPS ${base_test_DEPS} ARGS ${mem_opt} ${base_test_ARGS})
endfunction()
