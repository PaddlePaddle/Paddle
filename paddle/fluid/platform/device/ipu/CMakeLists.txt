IF(WITH_IPU)
  FILE(GLOB POPART_CANONICALIZATION_SRC ${PADDLE_SOURCE_DIR}/paddle/fluid/platform/device/ipu/popart_canonicalization/*.cc)
  list(APPEND PADDLE_IPU_SRC ${POPART_CANONICALIZATION_SRC})
  set(PADDLE_IPU_LIB "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_ipu.so" CACHE STRING "")
  set(PADDLE_IPU_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "")
  set(IPU_BACKEND_SRC
    "ipu_device.cc"
    "ipu_strategy.cc"
    "ipu_executor.cc"
    "ipu_compiler.cc"
    "ipu_backend.cc"
    "ipu_utils.cc"
  )

  cc_library(ipu_backend SRCS ${IPU_BACKEND_SRC} DEPS popart graph framework_proto enforce graph_helper timer)
  cc_library(ipu_info SRCS ipu_info.cc DEPS ipu_backend)
  cc_library(paddle_ipu SHARED SRCS ${PADDLE_IPU_SRC} DEPS popart)
  add_dependencies(paddle_ipu ipu_backend)
ENDIF()
