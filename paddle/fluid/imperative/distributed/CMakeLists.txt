cc_library(processgroup SRCS ProcessGroup.cc DEPS tensor)
cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS tensor)
cc_library(nccl_utils SRCS NCCLUtils.cc DEPS tensor enforce)
