if(WITH_CUDNN_FRONTEND)
  cc_library(
    cache
    SRCS cache.cc
    DEPS cudnn-frontend phi_enforce)
else()
  cc_library(
    cache
    SRCS cache.cc
    DEPS phi_enforce)
endif()
cc_library(
  switch_autotune
  SRCS switch_autotune.cc
  DEPS cache flags)
