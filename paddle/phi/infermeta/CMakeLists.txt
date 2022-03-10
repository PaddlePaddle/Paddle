cc_library(infermeta SRCS nullary.cc unary.cc binary.cc ternary.cc multiary.cc DEPS convert_utils meta_tensor infermeta_utils)
cc_library(backward_infermeta SRCS backward.cc DEPS meta_tensor convert_utils)
