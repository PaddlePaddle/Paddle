add_subdirectory(cpu)
if(WITH_GPU OR WITH_ROCM)
  add_subdirectory(cuda)
endif()
