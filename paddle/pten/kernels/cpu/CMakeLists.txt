cc_library(math_cpu SRCS math.cc DEPS dense_tensor kernel_context kernel_factory eigen_function blas pten_transpose_cpu)
cc_library(linalg_cpu SRCS linalg.cc DEPS dense_tensor kernel_context kernel_factory)
cc_library(manipulation_cpu SRCS manipulation.cc DEPS dense_tensor kernel_context kernel_factory copy_kernel unary)
