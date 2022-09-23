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
         dense_tensor)
  if(WITH_DISTRIBUTE AND WITH_PSCORE)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
      set_source_files_properties(
        ProcessGroupHeter.cc PROPERTIES COMPILE_FLAGS
                                        ${DISTRIBUTE_COMPILE_FLAGS})
    endif()
    cc_library(
      processgroup_heter
      SRCS ProcessGroupHeter.cc NCCLTools.cc Common.cc
      DEPS place enforce collective_helper device_context phi_api eager_api)
  endif()
endif()

if(WITH_MPI)
  cc_library(
    processgroup_mpi
    SRCS ProcessGroupMPI.cc MPITools.cc Common.cc
    DEPS collective_helper device_context)
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
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
      set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
      set_source_files_properties(
        ProcessGroupHeter.cc PROPERTIES COMPILE_FLAGS
                                        ${DISTRIBUTE_COMPILE_FLAGS})
    endif()

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
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    processgroup_custom
    SRCS ProcessGroupCustom.cc CustomCCLTools.cc Common.cc
    DEPS phi_backends
         place
         enforce
         collective_helper
         device_context
         phi_api
         eager_api)
endif()
