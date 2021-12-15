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
set(PTEN_DEPS convert_utils dense_tensor kernel_factory kernel_context)
set(PTEN_DEPS ${PTEN_DEPS} math_cpu linalg_cpu creation_cpu manipulation_cpu)
set(PTEN_DEPS ${PTEN_DEPS} nary unary binary)
if(WITH_GPU OR WITH_ROCM)
  set(PTEN_DEPS ${PTEN_DEPS} math_cuda linalg_cuda creation_cuda manipulation_cuda)
endif()
if(WITH_XPU)
  set(PTEN_DEPS ${PTEN_DEPS} manipulation_xpu)
endif()

cc_library(pten SRCS all.cc DEPS ${PTEN_DEPS})
