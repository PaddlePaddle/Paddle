if(WITH_XPU)
  cc_library(
    beam_search_xpu
    SRCS beam_search_xpu.cc
    DEPS xpu_op_list)
  math_library(beam_search DEPS phi common beam_search_xpu)
else()
  math_library(beam_search DEPS phi common)
endif()
