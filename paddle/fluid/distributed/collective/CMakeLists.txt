cc_library(
<<<<<<< HEAD
  process_group
  SRCS process_group.cc
  DEPS dense_tensor)

cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api process_group phi_api string_helper)

if(WITH_DISTRIBUTE)
  cc_library(
    process_group_gloo
    SRCS process_group_gloo.cc
    DEPS phi_api eager_api gloo_wrapper tcp_store)
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
<<<<<<< HEAD
    process_group_nccl
    SRCS process_group_nccl.cc nccl_tools.cc common.cc
    DEPS process_group
         tcp_store
=======
    processgroup_nccl
    SRCS ProcessGroupNCCL.cc NCCLTools.cc Common.cc
    DEPS processgroup
         processgroup_stream
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
         place
         enforce
         collective_helper
         device_context
<<<<<<< HEAD
         ${DEVICE_EVENT_LIBS}
         dense_tensor
         comm_static_check
         nccl_dynamic_check)
endif()

if(WITH_XPU_BKCL)
  cc_library(
    process_group_bkcl
    SRCS process_group_bkcl.cc bkcl_tools.cc common.cc
    DEPS process_group
         tcp_store
         place
         enforce
         collective_helper
         device_context
         dense_tensor)
endif()

if(WITH_MPI)
  cc_library(
    process_group_mpi
    SRCS process_group_mpi.cc mpi_tools.cc common.cc
    DEPS collective_helper device_context)
=======
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
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
<<<<<<< HEAD
    process_group_custom
    SRCS process_group_custom.cc custom_ccl_tools.cc common.cc
    DEPS process_group
         tcp_store
         phi_backends
         place
         enforce
         collective_helper
         device_context)
=======
    processgroup_custom
    SRCS ProcessGroupCustom.cc CustomCCLTools.cc Common.cc
    DEPS phi_backends
         place
         enforce
         collective_helper
         device_context
         phi_api
         eager_api)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
endif()
