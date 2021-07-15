add_subdirectory(user/src)

set(PTEN_DEPS base_tensor convert_utils)
if(WITH_GPU)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
