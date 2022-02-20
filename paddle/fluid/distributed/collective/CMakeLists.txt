cc_library(processgroup SRCS ProcessGroup.cc DEPS pten pten_api eager_api)

if(WITH_NCCL)
    cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context pten pten_api eager_api)
endif()
