cc_library(place SRCS place.cc)
cc_library(ddim SRCS ddim.cc)
cc_library(allocation SRCS allocation.cc malloc.cc default_allocator.cc DEPS place)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
