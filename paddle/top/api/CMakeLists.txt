add_subdirectory(src)

set(PTEN_DEPS convert_utils dense_tensor selected_rows_tensor eigen_function)
if(WITH_GPU)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda)
endif()

cc_library(top SRCS all.cc DEPS ${PTEN_DEPS})
