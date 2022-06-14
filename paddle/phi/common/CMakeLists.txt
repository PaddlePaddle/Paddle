cc_library(phi_place SRCS place.cc)
cc_library(
  scalar
  SRCS scalar.cc
  DEPS phi_enforce tensor)
cc_library(
  int_array
  SRCS int_array.cc
  DEPS phi_enforce tensor)
