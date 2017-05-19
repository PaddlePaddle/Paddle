cc_library(place SRCS place.cc)
cc_library(ddim SRCS ddim.cc)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
