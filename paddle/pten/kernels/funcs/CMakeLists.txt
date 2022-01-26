add_subdirectory(eigen)

cc_library(pten_transpose_cpu SRCS transpose.cc DEPS dense_tensor pten_context)
if(WITH_GPU)
  nv_library(pten_transpose_gpu SRCS transpose.cu DEPS dense_tensor malloc pten_context)
elseif(WITH_ROCM)
  hip_library(pten_transpose_gpu SRCS transpose.cu DEPS dense_tensor malloc pten_context)
endif()
