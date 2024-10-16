# GPU
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(gpu)
endif()

# IPU
if(WITH_IPU)
  add_subdirectory(ipu)
endif()
