cc_library(math_cpu SRCS math.cc DEPS dense_tensor kernel_context kernel_factory eigen_function)
cc_library(linalg_cpu SRCS linalg.cc DEPS dense_tensor kernel_context kernel_factory)
cc_library(fill_cpu SRCS fill.cc DEPS dense_tensor kernel_context kernel_factory eigen_function)
