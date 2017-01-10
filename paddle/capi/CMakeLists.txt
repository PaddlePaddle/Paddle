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

add_library(paddle_capi STATIC ${CAPI_SOURCES})

target_include_directories(paddle_capi PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_style_check_target(paddle_capi ${CAPI_SOURCES} ${CAPI_HEADER}
  ${CAPI_PRIVATE_HEADER})
add_dependencies(paddle_capi gen_proto_cpp)

set(PADDLE_CAPI_INC_PATH
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})

if (WITH_TESTING)
  add_subdirectory(tests)
endif()
