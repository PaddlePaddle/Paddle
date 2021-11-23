add_subdirectory(lib)

cc_library(pten_api SRCS all.cc DEPS linalg_api math_api creation_api manipulation_api utils_api)
