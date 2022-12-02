cc_library(
  processgroup
  SRCS ProcessGroup.cc
  DEPS dense_tensor)
cc_library(
  processgroup_stream
  SRCS ProcessGroupStream.cc
  DEPS dense_tensor)
cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api processgroup processgroup_stream phi_api string_helper)

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
endif()
