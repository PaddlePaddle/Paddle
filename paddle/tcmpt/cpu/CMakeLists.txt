cc_library(math_cpu SRCS math.cc DEPS dense_tensor kernel_context kernel_factory eigen_function)
cc_library(dot_cpu SRCS dot.cc DEPS dense_tensor kernel_context kernel_factory)
