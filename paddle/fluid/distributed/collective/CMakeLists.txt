cc_library(processgroup SRCS ProcessGroup.cc DEPS pten)
cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context pten)
