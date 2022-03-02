add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)

math_library(math_function DEPS blas dense_tensor tensor)
math_library(segment_pooling DEPS dense_tensor)
math_library(concat_and_split_functor DEPS dense_tensor)
