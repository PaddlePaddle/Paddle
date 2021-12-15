cc_library(utils_xpu SRCS utils.cc DEPS dense_tensor kernel_context kernel_factory memory convert_utils)
cc_library(manipulation_xpu SRCS manipulation.cc DEPS dense_tensor kernel_context kernel_factory utils_xpu unary)
