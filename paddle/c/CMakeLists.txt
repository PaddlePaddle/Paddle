cc_library(paddle_c SHARED SRCS scope.cc)
install(TARGETS paddle_c DESTINATION lib)
install(FILES error.h scope.h dll_exports.h
        DESTINATION include/paddle/v2)
