# XPU
IF(WITH_XPU)
  add_subdirectory(xpu)
ENDIF()
IF(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
ENDIF()

# NPU
IF(WITH_ASCEND OR WITH_ASCEND_CL)
  add_subdirectory(npu)
ENDIF()

# IPU
IF(WITH_IPU)
  add_subdirectory(ipu)
ENDIF()
