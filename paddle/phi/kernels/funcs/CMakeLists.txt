add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(lapack)
add_subdirectory(detail)

math_library(deformable_conv_functor DEPS dense_tensor)
math_library(concat_and_split_functor DEPS dense_tensor)
math_library(fc_functor DEPS blas jit_kernel_helper)
math_library(gpc DEPS phi_enforce)
math_library(gru_compute DEPS activation_functions math_function)
math_library(lstm_compute DEPS activation_functions)
math_library(math_function DEPS blas dense_tensor)
math_library(matrix_reduce DEPS dense_tensor)
math_library(matrix_inverse DEPS dense_tensor eigen3 blas)
math_library(pooling DEPS dense_tensor)
math_library(segment_pooling)
math_library(sequence2batch)
math_library(matrix_solve DEPS dense_tensor eigen3 blas math_function)

cc_library(
  phi_data_layout_transform
  SRCS data_layout_transform.cc
  DEPS tensor)

if(WITH_GPU OR WITH_ROCM)
  if(MKL_FOUND AND WITH_ONEMKL)
    math_library(fft spectral_op.cu DEPS dynload_cuda dynload_mklrt
                 dense_tensor)
    target_include_directories(fft PRIVATE ${MKL_INCLUDE})
  else()
    math_library(fft spectral_op.cu DEPS dynload_cuda dense_tensor pocketfft)
  endif()
else()
  if(MKL_FOUND AND WITH_ONEMKL)
    mathp_library(fft DEPS dynload_mklrt dense_tensor)
    target_include_directories(fft PRIVATE ${MKL_INCLUDE})
  else()
    math_library(fft DEPS dense_tensor pocketfft)
  endif()
endif()

if(WITH_MKLDNN)
  math_library(selected_rows_functor DEPS selected_rows_utils math_function
               blas mixed_vector)
else()
  math_library(selected_rows_functor DEPS selected_rows_utils math_function
               blas mixed_vector)
endif()
