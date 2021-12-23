IF(WITH_MKLDNN)
    set(MKLDNN_CTX_DEPS mkldnn)
ELSE()
    set(MKLDNN_CTX_DEPS)
ENDIF()

IF(WITH_GPU OR WITH_ROCM)
cc_library(cuda_stream SRCS cuda_stream.cc DEPS enforce boost ${MKLDNN_CTX_DEPS})
ENDIF()
