set(DEV_LIBS custom_device)

# GPU
if(WITH_GPU
   OR WITH_ROCM
   OR WITH_MUSA)
  add_subdirectory(gpu)
endif()

# XPU
if(WITH_XPU)
  add_subdirectory(xpu)
endif()

# IPU
if(WITH_IPU)
  add_subdirectory(ipu)
endif()

if(WITH_CUSTOM_DEVICE)
  add_subdirectory(custom)
endif()
