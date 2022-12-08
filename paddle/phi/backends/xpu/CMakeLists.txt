cc_library(
  phi_xpu_op_list
  SRCS xpu_op_list.cc xpu1_op_list.cc xpu2_op_list.cc
  DEPS glog)
