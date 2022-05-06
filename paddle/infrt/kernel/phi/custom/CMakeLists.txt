add_subdirectory(common)
add_subdirectory(cpu)

if (INFRT_WITH_GPU)
    add_subdirectory(gpu)
endif()
