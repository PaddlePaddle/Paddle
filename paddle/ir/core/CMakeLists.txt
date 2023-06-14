set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

if(WITH_SHARED_IR AND NOT WIN32)
  cc_library(ir SHARED SRCS ${IR_SRCS})
else()
  cc_library(ir SRCS ${IR_SRCS})
endif()
