add_subdirectory(eigen)
add_subdirectory(blas)
add_subdirectory(general)


if(WITH_GPU)
  nv_library(pten_transpose SRCS transpose.cc transpose.cu DEPS dense_tensor malloc pten_context)
elseif(WITH_ROCM)
  hip_library(pten_transpose SRCS transpose.cc transpose.cu DEPS dense_tensor malloc pten_context)
else()
  cc_library(pten_transpose SRCS transpose.cc DEPS dense_tensor pten_context)
endif()
