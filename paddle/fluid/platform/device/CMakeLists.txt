


IF(WITH_PLUGGABLE_DEVICE)
cc_library(callback_manager SRCS callback_manager.cc DEPS enforce place)

cc_library(device_guard SRCS device_guard.cc DEPS enforce place)

cc_library(stream SRCS stream.cc DEPS callback_manager enforce place)

cc_library(event SRCS event.cc DEPS enforce place)

cc_library(device_base SRCS device_base.cc DEPS stream event callback_manager device_guard enforce place device_context)

set(DEV_LIBS cpu_device)
cc_library(cpu_device SRCS cpu_device.cc DEPS stream event enforce place)

IF(WITH_GPU OR WITH_ROCM)
  cc_library(gpu_device SRCS gpu_device.cc DEPS stream event enforce place)
  set(DEV_LIBS ${DEV_LIBS} gpu_device)
ENDIF()

IF(WITH_ASCEND OR WITH_ASCEND_CL)
  cc_library(npu_device SRCS npu_device.cc DEPS stream event enforce place device_context)
  set(DEV_LIBS ${DEV_LIBS} npu_device)
ENDIF()

cc_library(plug_device SRCS plug_device.cc DEPS stream event enforce place device_context)
set(DEV_LIBS ${DEV_LIBS} plug_device)

cc_library(device_manager SRCS device_manager.cc DEPS ${DEV_LIBS} device_base)
set(GLOB_DEV_LIB ${DEV_LIBS} CACHE INTERNAL "Global DEV library")
ENDIF()

# GPU
IF(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
ENDIF()

# XPU
IF(WITH_XPU)
  add_subdirectory(xpu)
ENDIF()

# NPU
IF(WITH_ASCEND OR WITH_ASCEND_CL)
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
