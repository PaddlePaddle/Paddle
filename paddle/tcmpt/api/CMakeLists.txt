add_subdirectory(src)

set(TCMPT_DEPS convert_utils dense_tensor selected_rows_tensor kernel_factory kernel_context)
set(TCMPT_DEPS ${TCMPT_DEPS} math_cpu dot_cpu)
if(WITH_GPU OR WITH_ROCM)
  set(TCMPT_DEPS ${TCMPT_DEPS} math_cuda dot_cuda)
endif()

set(TCMPT_DEPS ${TCMPT_DEPS} math_api dot_api)

cc_library(tcmpt SRCS all.cc DEPS ${TCMPT_DEPS})
