cc_library(io SRCS io.cc DEPS proto_desc paddle_memory executor prune init)

if(NOT WITH_C_API AND WITH_FLUID)
  install(FILES io.h DESTINATION include/paddle/io)
endif()
