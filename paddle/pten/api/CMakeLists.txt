add_subdirectory(lib)

cc_library(pten_api SRCS all.cc DEPS pten_function_api utils_api)
