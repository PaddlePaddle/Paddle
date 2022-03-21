
set(DEV_LIBS custom_device)

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
