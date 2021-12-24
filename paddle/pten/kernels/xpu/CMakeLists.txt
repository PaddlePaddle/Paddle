cc_library(manipulation_xpu SRCS manipulation.cc DEPS dense_tensor kernel_context kernel_factory copy_kernel unary)
