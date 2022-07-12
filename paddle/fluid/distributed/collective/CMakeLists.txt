cc_library(
  processgroup
  SRCS ProcessGroup.cc
  DEPS)
cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS processgroup
       string_helper
       accumulation_node
       hook_utils
       tensor_utils
       autograd_meta
       utils
       tensor
       math_function
       string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    processgroup_gloo
    SRCS ProcessGroupGloo.cc
    DEPS gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    processgroup_nccl
    SRCS ProcessGroupNCCL.cc NCCLTools.cc Common.cc
    DEPS place
         cuda_stream
         enforce
         collective_helper
         device_context
         phi_api
         eager_api)
  if(WITH_DISTRIBUTE AND WITH_PSCORE)
    cc_library(
      processgroup_heter
      SRCS ProcessGroupHeter.cc NCCLTools.cc Common.cc
      DEPS place cuda_stream enforce collective_helper device_context)
  endif()
endif()

if(WITH_ASCEND_CL)
  cc_library(
    processgroup_hccl
    SRCS ProcessGroupHCCL.cc HCCLTools.cc Common.cc
    DEPS place
         npu_stream
         enforce
         collective_helper
         device_context
         phi_api
         eager_api)
  if(WITH_DISTRIBUTE AND WITH_PSCORE)
    cc_library(
      processgroup_heter
      SRCS ProcessGroupHeter.cc HCCLTools.cc Common.cc
      DEPS place npu_stream enforce collective_helper device_context)
  endif()
endif()
