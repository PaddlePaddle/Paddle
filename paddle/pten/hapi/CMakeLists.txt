add_subdirectory(lib)

cc_library(pten_hapi SRCS all.cc DEPS math_api linalg_api creation_api)
