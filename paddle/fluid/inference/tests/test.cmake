set(INFERENCE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
    "A path setting inference demo download directories.")
function (inference_download install_dir url filename)
    message(STATUS "Download inference test stuff from ${url}/${filename}")
    file(DOWNLOAD "${url}/${filename}" "${install_dir}/${filename}")
    message(STATUS "finish downloading ${filename}")
endfunction()

function (inference_download_and_uncompress install_dir url filename)
    inference_download(${install_dir} ${url} ${filename})
    execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${install_dir}/${filename}
            WORKING_DIRECTORY ${install_dir}
    )
endfunction()

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")
if (NOT EXISTS ${WORD2VEC_INSTALL_DIR})
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
