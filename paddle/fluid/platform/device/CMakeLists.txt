# XPU
IF(WITH_XPU)
  add_subdirectory(xpu)
ENDIF()

# NPU
IF(WITH_ASCEND OR WITH_ASCEND_CL)
  add_subdirectory(npu)
ENDIF()

# MLU
IF(WITH_MLU)
  add_subdirectory(mlu)
ENDIF()
