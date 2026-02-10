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

if(WITH_CUSTOM_DEVICE AND NOT WIN32)
  set(INIT_DEPS phi common custom_device_load)
else()
  set(INIT_DEPS phi common)
endif()

cc_library(
  init
  SRCS init.cc
  DEPS ${INIT_DEPS})

# Manage all device event library
set(DEVICE_EVENT_LIBS)

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
