cc_library(infermeta SRCS nullary.cc unary.cc binary.cc multiary.cc grad_infermeta.cc DEPS convert_utils)
cc_library(grad_infermeta SRCS grad_infermeta.cc DEPS convert_utils)
