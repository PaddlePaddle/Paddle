cc_library(majel SRCS place.cc ddim.cc)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
