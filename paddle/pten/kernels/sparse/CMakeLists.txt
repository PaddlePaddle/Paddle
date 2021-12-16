add_subdirectory(cpu)
if(WITH_GPU)
    add_subdirectory(cuda)
endif()
