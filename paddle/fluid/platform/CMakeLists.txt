if(WITH_ONEDNN)
  set(ONEDNN_CTX_DEPS onednn)
else()
  set(ONEDNN_CTX_DEPS)
endif()

add_subdirectory(device)

set(dgc_deps "")
if(WITH_DGC)
  set(dgc_deps dgc)
endif()

if(WITH_IPU)
  set(IPU_CTX_DEPS ipu_info)
else()
  set(IPU_CTX_DEPS)
endif()

if(WITH_GPU)
  nv_library(
    stream_callback_manager
    SRCS stream_callback_manager.cc
    DEPS simple_threadpool phi_core common)
endif()
if(WITH_ROCM)
  hip_library(
    stream_callback_manager
    SRCS stream_callback_manager.cc
    DEPS simple_threadpool phi_core common)
endif()

if(WITH_GPU OR WITH_ROCM)
  set(STREAM_CALLBACK_DEPS stream_callback_manager)
else()
  set(STREAM_CALLBACK_DEPS)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_context
    SRCS gloo_context.cc
    DEPS framework_proto gloo_wrapper phi_core common)
endif()

# separate init from device_context to avoid cycle dependencies
cc_library(
  init
  SRCS init.cc
  DEPS device_context phi_core common)

#fluid_memory depends on device_context, here add deps individually for
# avoiding cycle dependencies

cc_library(
  device_context
  SRCS device_context.cc
  DEPS simple_threadpool
       xxhash
       ${STREAM_CALLBACK_DEPS}
       eigen3
       framework_proto
       ${IPU_CTX_DEPS}
       ${ONEDNN_CTX_DEPS}
       ${dgc_deps}
       dlpack
       phi_core
       common)

cc_library(
  collective_helper
  SRCS collective_helper.cc
  DEPS framework_proto phi_core common)

# Manage all device event library
set(DEVICE_EVENT_LIBS)

cc_library(
  lodtensor_printer
  SRCS lodtensor_printer.cc
  DEPS phi_core
       common
       tensor
       scope
       lod_tensor
       variable_helper
       framework_proto)

add_subdirectory(profiler)

cc_library(
  init_phi
  SRCS init_phi.cc
  DEPS init)

if(TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()
