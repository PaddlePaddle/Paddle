cc_library(processgroup SRCS ProcessGroup.cc DEPS pten)

if(NOT WIN32)
    if(WITH_NCCL)
        cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context pten)
    endif()
endif()
