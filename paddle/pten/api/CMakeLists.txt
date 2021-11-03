add_subdirectory(lib)

cc_library(pten_hapi SRCS all.cc DEPS linalg_api math_api creation_api)
