cc_library(math_cpu SRCS math.cc DEPS dense_tensor kernel_context kernel_factory eigen_function blas pten_transpose_cpu cast_kernel)
