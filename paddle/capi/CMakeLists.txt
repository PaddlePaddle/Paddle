if (WITH_DOUBLE)
  set(PADDLE_FLOAT_TYPE double)
else ()
  set(PADDLE_FLOAT_TYPE float)
endif()

configure_file(config.h.in config.h @ONLY)

set(CAPI_HEADER
   PaddleCAPI.h)
set(CAPI_PRIVATE_HEADER
   PaddleCAPIPrivate.h)
file(GLOB CAPI_SOURCES *.cpp)

add_library(paddle_capi SHARED ${CAPI_SOURCES})

target_include_directories(paddle_capi PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(paddle_capi gen_proto_cpp)

target_link_libraries(paddle_capi
  "-Wl,-force_load"
  paddle_gserver
  "-Wl,-force_load"
  paddle_function
  paddle_pserver
  paddle_trainer_lib
  paddle_network
  paddle_math
  paddle_utils
  paddle_parameter
  paddle_proto
  paddle_cuda
  ${PROTOBUF_LIBRARY}
  ${LIBGLOG_LIBRARY}
  gflags
  ${CMAKE_THREAD_LIBS_INIT}
  ${CBLAS_LIBS}
  ${ZLIB_LIBRARIES}
  ${INTERAL_LIBS}
  ${CMAKE_DL_LIBS}
  ${PYTHON_LIBRARIES})

set(PADDLE_CAPI_INC_PATH
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

if (WITH_TESTING)
  add_subdirectory(tests)
endif()
