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
cc_library(
  init
  SRCS init.cc
  DEPS phi common)

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
  lodtensor_printer
  SRCS lodtensor_printer.cc
  DEPS phi common tensor scope lod_tensor variable_helper)

add_subdirectory(profiler)

cc_library(
  init_phi
  SRCS init_phi.cc
  DEPS init)

if(TENSORRT_FOUND)
  add_subdirectory(tensorrt)
endif()
