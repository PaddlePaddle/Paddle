if(WITH_PYTHON)
  cc_library(paddle_pybind SHARED
    SRCS pybind.cc exception.cc protobuf.cc const_value.cc
    DEPS pybind python backward proto_desc paddle_memory executor feed_fetch_method prune init
    ${GLOB_OP_LIB})
  if(NOT APPLE AND NOT ANDROID)
    target_link_libraries(paddle_pybind rt)
  endif(NOT APPLE AND NOT ANDROID)
endif(WITH_PYTHON)

if(WITH_DOC)
  cc_binary(print_operators_doc SRCS print_operators_doc.cc DEPS ${GLOB_OP_LIB})
endif(WITH_DOC)
