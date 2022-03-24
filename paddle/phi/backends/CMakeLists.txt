add_subdirectory(dynload)

add_subdirectory(cpu)

add_subdirectory(custom)

if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
endif()

if(WITH_XPU)
  add_subdirectory(xpu)
endif()

cc_library(phi_context SRCS all_context.cc DEPS device_context cpu_context)

if(WITH_XPU)
  add_dependencies(phi_context xpu_context)
endif()

if(WITH_GPU)
  add_dependencies(phi_context gpu_context)
endif()

if(WITH_CUSTOM_DEVICE)
  add_dependencies(phi_context custom_context)
  cc_library(callback_manager SRCS callback_manager.cc DEPS enforce place)
  cc_library(device_guard SRCS device_guard.cc DEPS enforce place)
  cc_library(stream SRCS stream.cc DEPS callback_manager)
  cc_library(event SRCS event.cc DEPS enforce place)
  cc_library(device_base SRCS device_base.cc DEPS stream event callback_manager device_guard device_context flags)
  cc_library(device_manager SRCS device_manager.cc DEPS custom_device)
  set(GLOB_DEV_LIB device_manager custom_device CACHE INTERNAL "Global DEV library")
endif()
