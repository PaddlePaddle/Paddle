file(GLOB DRR_SRCS "*.cc" "api/*.cc")

cc_library(
  drr
  SRCS ${DRR_SRCS}
  DEPS pd_dialect ir)
