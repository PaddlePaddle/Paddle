cc_library(processgroup SRCS ProcessGroup.cc DEPS phi phi_api eager_api)
if (WITH_DISTRIBUTE)
  cc_library(processgroup_gloo SRCS ProcessGroupGloo.cc DEPS phi phi_api eager_api gloo_wrapper)
endif()
cc_library(eager_reducer SRCS reducer.cc DEPS eager_api processgroup)

if(WITH_NCCL)
    cc_library(processgroup_nccl SRCS ProcessGroupNCCL.cc DEPS place cuda_stream enforce collective_helper device_context phi phi_api eager_api)
endif()
