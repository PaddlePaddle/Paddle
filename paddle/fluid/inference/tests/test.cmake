include(ExternalProject)
set(INFERENCE_URL "http://paddle-inference-dist.cdn.bcebos.com" CACHE STRING "inference download url")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
    "A path setting inference demo download directories.")

function(inference_download TARGET_NAME INSTALL_DIR URL FILENAME)
  if(NOT EXISTS ${INSTALL_DIR} AND NOT EXISTS ${INSTALL_DIR}/${FILENAME})
    message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
    string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
    set(DOWNLOAD_CMAKE_FILENAME "download_${FILENAME_EX}.cmake")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${DOWNLOAD_CMAKE_FILENAME}
"message(STATUS \"Downloading...
         src='${URL}/${FILENAME}'
         dst='${INSTALL_DIR}/${FILENAME}'\")

file(DOWNLOAD
  \"${URL}/${FILENAME}\"
  \"${INSTALL_DIR}/${FILENAME}\"
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR \"Error: downloading '${remote}' failed
                      status_code: \${status_code}
                      status_string: \${status_string}
                      log: \${log}\")
endif()

message(STATUS \"Downloading ${FILENAME} ... done\")
"
)

    add_custom_target(download_${FILENAME_EX})
    add_custom_command(
        TARGET download_${FILENAME_EX}
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/${DOWNLOAD_CMAKE_FILENAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    set(${TARGET_NAME} "${${TARGET_NAME}} download_${FILENAME_EX}" PARENT_SCOPE)
  endif()
endfunction()

function(inference_download_and_uncompress TARGET_NAME INSTALL_DIR URL FILENAME)
  if(NOT EXISTS ${INSTALL_DIR})
    inference_download(${TARGET_NAME} ${INSTALL_DIR} ${URL} ${FILENAME})

    if(FILENAME MATCHES "(\\.|=)(7z|tar\\.bz2|tar\\.gz|tar\\.xz|tbz2|tgz|txz|zip)$")
      string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
      add_custom_target(uncompress_${FILENAME_EX} DEPENDS download_${FILENAME_EX})
      add_custom_command(
          TARGET uncompress_${FILENAME_EX}
          COMMAND ${CMAKE_COMMAND} -E tar xfz ${INSTALL_DIR}/${FILENAME}
          COMMENT "-- Uncompressing ${FILENAME} ..."
          WORKING_DIRECTORY ${INSTALL_DIR})

      set(${TARGET_NAME} "${${TARGET_NAME}} uncompress_${FILENAME_EX}" PARENT_SCOPE)
    else()
      set(${TARGET_NAME} "${${TARGET_NAME}}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(add_download_dependencies TARGET DOWNLOAD_DEPS)
  string(REPLACE " " ";" DEP_LIST ${DOWNLOAD_DEPS})
  foreach(dep ${DEP_LIST})
    add_dependencies(${TARGET} ${dep})
  endforeach()
endfunction()

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")
if(NOT EXISTS ${WORD2VEC_INSTALL_DIR} AND NOT WIN32)
  inference_download_and_uncompress(download_word2vec ${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL} "word2vec.inference.model.tar.gz")
  set(WORD2VEC_DOWNLOAD_DEPS ${download_word2vec})
endif()
set(WORD2VEC_MODEL_DIR "${WORD2VEC_INSTALL_DIR}/word2vec.inference.model")

function(inference_base_test TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS ARGS DEPS DOWNLOAD_DEPS)
  cmake_parse_arguments(base_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(WITH_GPU)
    set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
  endif()
  cc_test(${TARGET} SRCS ${base_test_SRCS} DEPS ${base_test_DEPS} ARGS ${mem_opt} ${base_test_ARGS})
  # download_deps can be added to DEPS, because they are not library targets.
  if(base_test_DOWNLOAD_DEPS)
    add_download_dependencies(${TARGET} ${base_test_DOWNLOAD_DEPS})
  endif()
endfunction()
