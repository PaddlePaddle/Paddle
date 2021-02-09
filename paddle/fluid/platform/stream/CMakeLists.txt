IF(WITH_GPU)
cc_library(cuda_stream SRCS cuda_stream.cc DEPS enforce boost stream_callback_manager) 
ENDIF()

IF(WITH_ASCEND_CL)
cc_library(npu_stream SRCS npu_stream.cc DEPS enforce boost stream_callback_manager)
ENDIF()
