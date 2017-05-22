cc_library(place SRCS place.cc)
cc_library(ddim SRCS ddim.cc)
cc_library(malloc SRCS malloc.cc)
cc_library(allocation SRCS allocation.cc)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
