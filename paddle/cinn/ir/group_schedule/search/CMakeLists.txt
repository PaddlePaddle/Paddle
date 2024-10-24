core_gather_headers()

cc_library(
  schedule_config_search
  SRCS config_searcher.cc measurer.cc
  DEPS add_cinn_pass)
