set(TCMPT_DEPS convert_utils dense_tensor kernel_factory kernel_context)
set(TCMPT_DEPS ${TCMPT_DEPS} math_cpu linalg_cpu creation_cpu manipulation_cpu)
set(TCMPT_DEPS ${TCMPT_DEPS} unary binary)
if(WITH_GPU OR WITH_ROCM)
  set(TCMPT_DEPS ${TCMPT_DEPS} math_cuda linalg_cuda creation_cuda manipulation_cuda)
endif()

cc_library(tcmpt SRCS all.cc DEPS ${TCMPT_DEPS})
