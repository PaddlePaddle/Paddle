cc_library(
  sparse_infermeta
  SRCS unary.cc binary.cc multiary.cc
  DEPS convert_utils infermeta_utils)

cc_library(
  sparse_backward_infermeta
  SRCS backward.cc
  DEPS meta_tensor convert_utils)
