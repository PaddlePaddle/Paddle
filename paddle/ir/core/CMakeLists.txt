set(NEWIR_SOURCE_DIR "${PADDLE_SOURCE_DIR}/paddle/ir")
set(NEWIR_BINARY_DIR "${PADDLE_BINARY_DIR}/paddle/ir")

file(GLOB IR_SRCS "*.cc")

if(WITH_SHARED_IR)
  cc_library(ir SHARED SRCS ${IR_SRCS})
else()
  cc_library(ir SRCS ${IR_SRCS})
endif()

# if(WIN32)
#   set_property(TARGET ir PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
# endif()
