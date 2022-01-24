cc_library(infermeta SRCS nullary.cc unary.cc binary.cc multiary.cc DEPS convert_utils meta_tensor)
cc_library(backward_infermeta SRCS backward.cc DEPS convert_utils meta_tensor)
