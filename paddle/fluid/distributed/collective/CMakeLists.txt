cc_library(
  processgroup
  SRCS ProcessGroup.cc
<<<<<<< HEAD
  DEPS phi_api eager_api)
cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api processgroup phi_api string_helper)
=======
  DEPS dense_tensor)
cc_library(
  processgroup_stream
  SRCS ProcessGroupStream.cc
  DEPS dense_tensor)
cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api processgroup processgroup_stream phi_api string_helper)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

if(WITH_DISTRIBUTE)
  cc_library(
    processgroup_gloo
    SRCS ProcessGroupGloo.cc
    DEPS phi_api eager_api gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    processgroup_nccl
    SRCS ProcessGroupNCCL.cc NCCLTools.cc Common.cc
<<<<<<< HEAD
    DEPS place enforce collective_helper device_context phi_api eager_api)
  if(WITH_DISTRIBUTE AND WITH_PSCORE)
    cc_library(
      processgroup_heter
      SRCS ProcessGroupHeter.cc NCCLTools.cc Common.cc
      DEPS place enforce collective_helper device_context phi_api eager_api)
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
      DEPS place
           npu_stream
           enforce
           collective_helper
           device_context
           phi_api
           eager_api)
  endif()
=======
    DEPS processgroup
         processgroup_stream
         place
         enforce
         collective_helper
         device_context
         ${DEVICE_EVENT_LIBS}
         dense_tensor)
endif()

if(WITH_XPU_BKCL)
  cc_library(
    processgroup_bkcl
    SRCS ProcessGroupBKCL.cc BKCLTools.cc Common.cc
    DEPS processgroup place enforce collective_helper device_context
         dense_tensor)
endif()

if(WITH_MPI)
  cc_library(
    processgroup_mpi
    SRCS ProcessGroupMPI.cc MPITools.cc Common.cc
    DEPS collective_helper device_context)
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    processgroup_custom
    SRCS ProcessGroupCustom.cc CustomCCLTools.cc Common.cc
    DEPS processgroup phi_backends place enforce collective_helper
         device_context)
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
endif()
