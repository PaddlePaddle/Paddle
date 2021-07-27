add_subdirectory(src)

set(PTEN_DEPS convert_utils dense_tensor selected_rows_tensor)
if(WITH_GPU)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
