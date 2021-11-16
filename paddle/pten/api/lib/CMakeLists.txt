add_subdirectory(utils)

cc_library(pten_device_context_pool SRCS device_context_pool.cc DEPS pten)
cc_library(math_api SRCS math.cc DEPS pten)
cc_library(linalg_api SRCS linalg.cc DEPS pten)
cc_library(creation_api SRCS creation.cc DEPS pten)
cc_library(manipulation_api SRCS manipulation.cc DEPS pten)
