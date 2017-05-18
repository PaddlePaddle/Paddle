cc_library(majel SRCS place.cc allocation.cc)

if(WITH_TESTING)
    add_subdirectory(test)
endif()
