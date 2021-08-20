if(NOT WITH_MKL)
    return()
endif()

if(WITH_ONEMKL)
    find_package(MKL REQUIRED)
endif()
