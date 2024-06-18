file(GLOB dist_tensor_spec_srcs *.cc)

cc_library(
  dist_tensor_spec
  SRCS ${dist_tensor_spec_srcs}
  DEPS phi common)
