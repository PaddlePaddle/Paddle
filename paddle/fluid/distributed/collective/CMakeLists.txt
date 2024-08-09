cc_library(
  eager_reducer
  SRCS reducer.cc
  DEPS eager_api phi common string_helper)
