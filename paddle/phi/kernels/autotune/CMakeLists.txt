cc_library(cache SRCS cache.cc)
cc_library(
  switch_autotune
  SRCS switch_autotune.cc
  DEPS cache flags)
