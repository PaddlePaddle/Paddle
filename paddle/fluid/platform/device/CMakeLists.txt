set(DEV_LIBS custom_device)

# XPU
if(WITH_XPU)
  add_subdirectory(xpu)
endif()

# IPU
if(WITH_IPU)
  add_subdirectory(ipu)
endif()
