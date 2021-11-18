# set(eager_deps autograd_meta grad_node_info pten pten_api)

add_subdirectory(tests)

# cc_library(autograd_meta SRCS autograd_meta.cc DEPS pten pten_api)
# cc_library(utils SRCS utils.cc DEPS pten pten_api layer proto_desc operator op_registry variable_helper memcpy)
# cc_library(grad_node_info SRCS grad_node_info.cc DEPS pten pten_api)
