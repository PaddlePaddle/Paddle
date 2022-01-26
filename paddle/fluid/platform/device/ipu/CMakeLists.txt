IF(WITH_IPU)
  FILE(GLOB POPART_CANONICALIZATION_SRC ${PADDLE_SOURCE_DIR}/paddle/fluid/platform/device/ipu/popart_canonicalization/*.cc)
  list(APPEND PADDLE_IPU_SRC ${POPART_CANONICALIZATION_SRC})
  set(IPU_BACKEND_SRC
    "ipu_strategy.cc"
    "ipu_executor.cc"
    "ipu_compiler.cc"
    "ipu_backend.cc"
    "ipu_utils.cc"
  )
  set(IPU_INFO_SRC
    "ipu_info.cc"
    "ipu_device.cc"
  )

  cc_library(ipu_backend SRCS ${IPU_BACKEND_SRC} DEPS popart graph graph_helper)
  cc_library(ipu_info SRCS ${IPU_INFO_SRC} DEPS popart enforce)
  cc_library(paddle_ipu SHARED SRCS ${PADDLE_IPU_SRC} DEPS popart graph_helper)
  add_dependencies(paddle_ipu ipu_backend)
  set(PADDLE_IPU_LIB "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_ipu.so" CACHE STRING "")
  set(PADDLE_IPU_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE STRING "")
ENDIF()
