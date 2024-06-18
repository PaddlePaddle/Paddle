include(test.cmake) # some generic cmake function for inference

set(WORD2VEC_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/word2vec")

if(NOT EXISTS ${WORD2VEC_INSTALL_DIR}/word2vec.inference.model.tar.gz)
  inference_download_and_uncompress_without_verify(
    ${WORD2VEC_INSTALL_DIR} ${INFERENCE_URL} "word2vec.inference.model.tar.gz")
endif()

set(WORD2VEC_MODEL_DIR "${WORD2VEC_INSTALL_DIR}/word2vec.inference.model")

set(IMG_CLS_RESNET_INSTALL_DIR
    "${INFERENCE_DEMO_INSTALL_DIR}/image_classification_resnet")

if(NOT EXISTS
   ${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model.tgz
)
  inference_download_and_uncompress_without_verify(
    ${IMG_CLS_RESNET_INSTALL_DIR} ${INFERENCE_URL}
    "image_classification_resnet.inference.model.tgz")
endif()

set(IMG_CLS_RESNET_MODEL_DIR
    "${IMG_CLS_RESNET_INSTALL_DIR}/image_classification_resnet.inference.model")

if(WITH_ONNXRUNTIME)
  set(MOBILENETV2_INSTALL_DIR "${INFERENCE_DEMO_INSTALL_DIR}/MobileNetV2")
  if(NOT EXISTS ${MOBILENETV2_INSTALL_DIR}/MobileNetV2.inference.model.tar.gz)
    inference_download_and_uncompress_without_verify(
      ${MOBILENETV2_INSTALL_DIR} ${INFERENCE_URL}
      "MobileNetV2.inference.model.tar.gz")
  endif()
  set(MOBILENETV2_MODEL_DIR "${MOBILENETV2_INSTALL_DIR}/MobileNetV2")
endif()
