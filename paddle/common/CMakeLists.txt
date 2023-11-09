file(GLOB common_srcs "*.cc")

set(COMMON_BUILD_TYPE
    SHARED
    CACHE INTERNAL "" FORCE)

cc_library(common SHARED SRCS ${common_srcs})
