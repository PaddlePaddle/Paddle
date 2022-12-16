set(DEV_LIBS custom_device)

# GPU
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
endif()

# XPU
if(WITH_XPU)
  add_subdirectory(xpu)
endif()

# NPU
if(WITH_ASCEND OR WITH_ASCEND_CL)
  add_subdirectory(npu)
endif()

# IPU
if(WITH_IPU)
  add_subdirectory(ipu)
endif()

# MLU
if(WITH_MLU)
  add_subdirectory(mlu)
endif()

if(WITH_CUSTOM_DEVICE)
  add_subdirectory(custom)
endif()
