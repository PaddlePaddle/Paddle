add_subdirectory(src)

set(TOP_DEPS convert_utils dense_tensor selected_rows_tensor kernel_factory kernel_context)
set(TOP_DEPS ${TOP_DEPS} math_cpu)
if(WITH_GPU)
  set(TOP_DEPS ${TOP_DEPS} math_cuda)
endif()
if(WITH_XPU)
  set(TOP_DEPS ${TOP_DEPS} math_xpu)
endif()

cc_library(top SRCS all.cc DEPS ${TOP_DEPS})
