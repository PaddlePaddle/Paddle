cc_library(processgroup SRCS ProcessGroup.cc DEPS phi phi_api eager_api)
cc_library(eager_reducer SRCS reducer.cc DEPS eager_api processgroup phi phi_api string_helper)

if (WITH_DISTRIBUTE)
  cc_library(processgroup_gloo SRCS ProcessGroupGloo.cc DEPS phi phi_api eager_api gloo_wrapper)
endif()

if(WITH_NCCL)
    cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context phi phi_api eager_api)
endif()
if(WITH_ASCEND_CL)
    cc_library(processgroup_hccl SRCS ProcessGroupHCCL.cc DEPS place npu_stream enforce collective_helper device_context phi phi_api eager_api)
endif()
