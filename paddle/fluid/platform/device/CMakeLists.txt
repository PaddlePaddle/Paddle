set(DEV_LIBS custom_device)

# GPU
if(WITH_GPU OR WITH_ROCM)
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
