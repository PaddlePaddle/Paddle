add_subdirectory(lib)

cc_library(tcmpt_hapi SRCS all.cc DEPS math_api linalg_api creation_api)
