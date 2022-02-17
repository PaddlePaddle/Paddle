add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)

math_library(math_function DEPS blas dense_tensor tensor)
