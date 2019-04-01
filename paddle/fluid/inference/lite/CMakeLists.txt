set(PADDLE_LITE_LIB paddle-lite)
set(PADDLE_LITE_SRCS test.cc)

if (BUILD_SHARED_LIBS)
  add_library(${PADDLE_LITE_LIB} SHARED ${PADDLE_LITE_SRCS})
else()
  add_library(${PADDLE_LITE_LIB} STATIC ${PADDLE_LITE_SRCS})
endif(BUILD_SHARED_LIBS)
#target_link_libraries(${PADDLE_LITE_LIB} )
