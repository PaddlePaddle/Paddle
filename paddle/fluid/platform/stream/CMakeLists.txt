IF(WITH_GPU OR WITH_ROCM)
    cc_library(cuda_stream SRCS cuda_stream.cc DEPS enforce boost eigen3 ${MKLDNN_CTX_DEPS})
ENDIF()
