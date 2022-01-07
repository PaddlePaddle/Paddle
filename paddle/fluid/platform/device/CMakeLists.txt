IF(WITH_CUSTOM_DEVICE)
cc_library(callback_manager SRCS callback_manager.cc DEPS enforce place)

cc_library(device_guard SRCS device_guard.cc DEPS enforce place)

cc_library(stream SRCS stream.cc DEPS callback_manager)

cc_library(event SRCS event.cc DEPS enforce place)

cc_library(device_base SRCS device_base.cc DEPS stream event callback_manager device_guard device_context)

cc_library(cpu_device SRCS cpu_device.cc DEPS device_base)
ENDIF()

set(DEV_LIBS cpu_device custom_device)

# GPU
IF(WITH_GPU OR WITH_ROCM)
  set(DEV_LIBS ${DEV_LIBS} gpu_device)
  add_subdirectory(gpu)
ENDIF()

# XPU
IF(WITH_XPU)
  add_subdirectory(xpu)
ENDIF()

# NPU
IF(WITH_ASCEND OR WITH_ASCEND_CL)
  set(DEV_LIBS ${DEV_LIBS} npu_device)
  add_subdirectory(npu)
ENDIF()

# IPU
IF(WITH_IPU)
  add_subdirectory(ipu)
ENDIF()

# MLU
IF(WITH_MLU)
  add_subdirectory(mlu)
ENDIF()

# CUSTOM
IF(WITH_CUSTOM_DEVICE)
  add_subdirectory(custom)

  cc_library(device_manager SRCS device_manager.cc DEPS ${DEV_LIBS})
  set(GLOB_DEV_LIB device_manager ${DEV_LIBS} CACHE INTERNAL "Global DEV library")
ENDIF()
