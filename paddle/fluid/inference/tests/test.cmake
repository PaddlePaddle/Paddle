include(ExternalProject)
set(MODEL_MD5_SET)
list(APPEND MODEL_MD5_SET transformer_prune.tar.gz)
list(APPEND MODEL_MD5_SET conv_bn_swish_split_gelu.tar.gz)
list(APPEND MODEL_MD5_SET lac_data.txt.tar.gz)
list(APPEND MODEL_MD5_SET lac_model.tar.gz)
list(APPEND MODEL_MD5_SET text_classification_data.txt.tar.gz)
list(APPEND MODEL_MD5_SET text-classification-Senta.tar.gz)
list(APPEND MODEL_MD5_SET ernie_model_4.tar.gz)
list(APPEND MODEL_MD5_SET ernie_model_4_unserialized.tgz)
list(APPEND MODEL_MD5_SET ernie_model_4_fp16_unserialized.tgz)
list(APPEND MODEL_MD5_SET Ernie_model.tar.gz)
list(APPEND MODEL_MD5_SET Ernie_data.txt.tar.gz)
list(APPEND MODEL_MD5_SET Ernie_result.txt.tar.gz)
list(APPEND MODEL_MD5_SET Ernie_large_model.tar.gz)
list(APPEND MODEL_MD5_SET Ernie_large_data.txt.tar.gz)
list(APPEND MODEL_MD5_SET Ernie_large_result.txt.tar.gz)
list(APPEND MODEL_MD5_SET trt_inference_test_models.tar.gz)

set(transformer_prune.tar.gz 77b56dc73ff0cf44ddb1ce9ca0b0f471)
set(conv_bn_swish_split_gelu.tar.gz 2a5e8791e47b221b4f782151d76da9c6)
set(lac_data.txt.tar.gz 9983539cd6b34fbdc411e43422776bfd)
set(lac_model.tar.gz 419ca6eb85f57a01bfe173591910aec5)
set(text_classification_data.txt.tar.gz 36ae620020cc3377f45ed330dd36238f)
set(text-classification-Senta.tar.gz 3f0f440313ca50e26184e65ffd5809ab)
set(ernie_model_4.tar.gz 5fa371efa75706becbaad79195d2ca68)
set(ernie_model_4_unserialized.tgz 833d73fc6a7f7e1ee4a1fd6419209e55)
set(ernie_model_4_fp16_unserialized.tgz c5ff2d0cad79953ffbf2b8b9e2fae6e4)
set(Ernie_model.tar.gz aa59192dd41ed377f9f168e3a1309fa6)
set(Ernie_data.txt.tar.gz 5396e63548edad7ca561e7e26a9476d1)
set(Ernie_result.txt.tar.gz 73beea65abda2edb61c1662cd3180c62)
set(Ernie_large_model.tar.gz af7715245ed32cc77374625d4c80f7ef)
set(Ernie_large_data.txt.tar.gz edb2113eec93783cad56ed76d47ba57f)
set(Ernie_large_result.txt.tar.gz 1facda98eef1085dc9d435ebf3f23a73)
set(trt_inference_test_models.tar.gz 3dcccdc38b549b6b1b4089723757bd98)

set(INFERENCE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")
set(INFERENCE_DEMO_INSTALL_DIR "${THIRD_PARTY_PATH}/inference_demo" CACHE STRING
    "A path setting inference demo download directories.")
set(CPU_NUM_THREADS_ON_CI 4 CACHE STRING "Run multi-threads on CI to reduce CI time.")
set(WARMUP_BATCH_SIZE 100 CACHE STRING "Default warmup_batch_size.")

function(inference_download INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  ExternalProject_Add(
      extern_inference_download_${FILENAME_EX}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${INSTALL_DIR}
      URL                   ${URL}/${FILENAME}
      DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME}
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
  string(REGEX REPLACE "[-%./\\]" "_" FILENAME_EX ${FILENAME})
  string(REGEX MATCH "[^/\\]+$" DOWNLOAD_NAME ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")

  list(FIND MODEL_MD5_SET ${FILENAME} MODEL_VER)
  if (${MODEL_VER} STREQUAL -1)
      set(model_md5 )
  else()
      list(GET ${FILENAME} 0 model_md5)
  endif()
  ExternalProject_Add(
      ${EXTERNAL_PROJECT_NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${INSTALL_DIR}
      URL                   ${URL}/${FILENAME}
      URL_MD5               ${model_md5}
      DOWNLOAD_DIR          ${INSTALL_DIR}
      DOWNLOAD_NO_EXTRACT   1
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ${CMAKE_COMMAND} -E chdir ${INSTALL_DIR}
                            ${CMAKE_COMMAND} -E tar xzf ${DOWNLOAD_NAME}
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
  )
endfunction()

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")
if(NOT EXISTS ${WORD2VEC_INSTALL_DIR}/word2vec.inference.model.tar.gz)
  inference_download_and_uncompress(${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL} "word2vec.inference.model.tar.gz")
endif()
set(WORD2VEC_MODEL_DIR "${WORD2VEC_INSTALL_DIR}/word2vec.inference.model")

set(IMG_CLS_RESNET_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/image_classification_resnet")
if(NOT EXISTS ${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model.tgz)
  inference_download_and_uncompress(${IMG_CLS_RESNET_INSTALL_DIR} ${INFERENCE_URL} "image_classification_resnet.inference.model.tgz")
endif()
set(IMG_CLS_RESNET_MODEL_DIR "${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model")

function (inference_base_test_build TARGET)
   set(options "")
   set(oneValueArgs "")
   set(multiValueArgs SRCS DEPS)
   cmake_parse_arguments(base_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
   cc_test_build(${TARGET} SRCS ${base_test_SRCS} DEPS ${base_test_DEPS})
endfunction()

function (inference_base_test_run TARGET)
   set(options "")
   set(oneValueArgs "")
   set(multiValueArgs COMMAND ARGS)
   cmake_parse_arguments(base_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
   if(WITH_GPU)
       set(mem_opt "--fraction_of_gpu_memory_to_use=0.5")
   endif()
   cc_test_run(${TARGET} COMMAND ${base_test_COMMAND} ARGS ${mem_opt} ${base_test_ARGS})
endfunction()

function (inference_base_test TARGET)
   set(options "")
   set(oneValueArgs "")
   set(multiValueArgs SRCS ARGS DEPS)
   cmake_parse_arguments(base_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
   inference_base_test_build(${TARGET}
	   SRCS ${base_test_SRCS}
	   DEPS ${base_test_DEPS})
   inference_base_test_run(${TARGET}
	   COMMAND ${TARGET}
	   ARGS ${base_test_ARGS})
endfunction()
