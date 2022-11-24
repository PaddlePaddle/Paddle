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
<<<<<<< HEAD
=======

if(WITH_CUSTOM_DEVICE)
  add_subdirectory(custom)
endif()
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
