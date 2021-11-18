add_subdirectory(tests)

set(eager_deps pten pten_api)

cc_library(grad_node_info SRCS grad_node_info.cc DEPS ${eager_deps})
