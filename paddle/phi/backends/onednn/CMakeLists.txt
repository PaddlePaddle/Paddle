cc_library(
  onednn_context
  SRCS onednn_context.cc
  DEPS cpu_context mkldnn)
