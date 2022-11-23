if(WITH_CUDNN_FRONTEND)
  cc_library(
    cache
    SRCS cache.cc
    DEPS cudnn-frontend)
else()
  cc_library(cache SRCS cache.cc)
endif()
cc_library(
  switch_autotune
  SRCS switch_autotune.cc
  DEPS cache flags)
