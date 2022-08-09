if(WITH_GPU)
  nv_library(
    phi_place
    SRCS place.cc
    DEPS phi_backends)
elseif(WITH_ROCM)
  hip_library(
    phi_place
    SRCS place.cc
    DEPS phi_backends)
else()
  cc_library(phi_place SRCS place.cc)
endif()

cc_library(
  scalar
  SRCS scalar.cc
  DEPS phi_enforce tensor)
cc_library(
  int_array
  SRCS int_array.cc
  DEPS phi_enforce tensor)
