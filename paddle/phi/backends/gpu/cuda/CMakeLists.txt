collect_srcs(backends_srcs SRCS cudnn_workspace_helper.cc)

if(WITH_GPU)
  nv_library(
    gpu_event_timer
    SRCS gpu_event_timer.cc
    DEPS phi_core glog)
endif()
