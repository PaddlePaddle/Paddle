add_subdirectory(device)

if(WITH_IPU)
  set(IPU_CTX_DEPS ipu_info)
else()
  set(IPU_CTX_DEPS)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_context
    SRCS gloo_context.cc
    DEPS gloo_wrapper phi common)
endif()

# separate init from device_context to avoid cycle dependencies
if(WITH_CUSTOM_DEVICE AND NOT WIN32)
  set(INIT_DEPS phi common custom_device_load)
else()
  set(INIT_DEPS phi common)
endif()

cc_library(
  init
  SRCS init.cc
  DEPS ${INIT_DEPS})

#fluid_memory depends on device_context, here add deps individually for
# avoiding cycle dependencies

cc_library(
  device_context
  SRCS device_context.cc
  DEPS simple_threadpool
       xxhash
       eigen3
       ${IPU_CTX_DEPS}
       dlpack
       phi
       common)

# Manage all device event library
set(DEVICE_EVENT_LIBS)
cc_library(
  device_event_base
  SRCS device_event_base.cc device_event_defs.cc
  DEPS device_context op_registry phi common)
set(DEVICE_EVENT_LIBS
    device_event_base
    CACHE INTERNAL "device event libs")
if(WITH_GPU)
  nv_library(
    device_event_gpu
    SRCS device_event_gpu.cc
    DEPS device_event_base)
  set(DEVICE_EVENT_LIBS
      device_event_gpu
      CACHE INTERNAL "device event libs")
endif()

if(WITH_ROCM)
  hip_library(
    device_event_gpu
    SRCS device_event_gpu.cc
    DEPS device_event_base)
  set(DEVICE_EVENT_LIBS
      device_event_gpu
      CACHE INTERNAL "device event libs")
endif()

if(WITH_CUSTOM_DEVICE)
  cc_library(
    device_event_custom_device
    SRCS device_event_custom_device.cc
    DEPS device_event_base phi)
  set(DEVICE_EVENT_LIBS
      ${DEVICE_EVENT_LIBS} device_event_custom_device
      CACHE INTERNAL "device event libs")
endif()

if(WITH_GPU)
  if(WITH_CUSTOM_DEVICE)
    nv_library(
      cuda_graph_with_memory_pool
      SRCS cuda_graph_with_memory_pool.cc
      DEPS ${DEVICE_EVENT_LIBS} device_event_custom_device device_context phi
           common)
  else()
    nv_library(
      cuda_graph_with_memory_pool
      SRCS cuda_graph_with_memory_pool.cc
      DEPS ${DEVICE_EVENT_LIBS} device_context phi common)
  endif()
else()
  cc_library(
    cuda_graph_with_memory_pool
    SRCS cuda_graph_with_memory_pool.cc
    DEPS device_context phi common)
endif()

cc_library(
  densetensor_printer
  SRCS densetensor_printer.cc
  DEPS phi common tensor scope lod_tensor variable_helper)

add_subdirectory(profiler)

cc_library(
  init_phi
  SRCS init_phi.cc
  DEPS init)

if(TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()
