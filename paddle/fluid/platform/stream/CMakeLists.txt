IF(WITH_GPU)
cc_library(cuda_stream SRCS cuda_stream.cc DEPS enforce boost)
ENDIF()
