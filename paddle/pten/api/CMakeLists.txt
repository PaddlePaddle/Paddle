add_subdirectory(user/src)

set(PTEN_DEPS convert_utils base_tensor pten_lod_tensor pten_selected_rows)
if(WITH_GPU)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
