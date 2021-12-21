# Adapt to custom op mechanism: Include the header files related to the data type
# to avoid exposing the path of the underlying file, remove it after moving
# float16.h/complex.h/bfloat16.h into pten
include_directories(${PADDLE_SOURCE_DIR}/paddle/fluid/platform)

# paddle experimental common components
add_subdirectory(common)

# pten (low level) api headers: include
# pten (high level) api
add_subdirectory(api)
# pten core components
add_subdirectory(core)
# pten components of specific backends
add_subdirectory(backends)
# pten kernels for diff device
add_subdirectory(kernels)
# pten infermeta
add_subdirectory(infermeta)
# pten operator definitions
add_subdirectory(ops)
# pten tests
add_subdirectory(tests)

# make an unity target for compile deps
set(PTEN_DEPS convert_utils dense_tensor pten_context kernel_factory kernel_context)
set(PTEN_DEPS ${PTEN_DEPS} math_cpu linalg_cpu manipulation_cpu conj_kernel_cpu scale_kernel_cpu full_kernel_cpu empty_kernel_cpu)
set(PTEN_DEPS ${PTEN_DEPS} nary unary binary)
if(WITH_GPU OR WITH_ROCM)
  set(PTEN_DEPS ${PTEN_DEPS} math_gpu linalg_gpu manipulation_gpu conj_kernel_gpu scale_kernel_gpu full_kernel_gpu empty_kernel_gpu)
endif()
if(WITH_XPU)
  set(PTEN_DEPS ${PTEN_DEPS} manipulation_xpu)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
