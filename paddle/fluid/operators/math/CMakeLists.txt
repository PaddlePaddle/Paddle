if(WITH_XPU)
  cc_library(
    beam_search_xpu
    SRCS beam_search_xpu.cc
    DEPS xpu_op_list)
endif()

# please add new math_library in alphabetical order
math_library(concat_and_split DEPS concat_and_split_functor)
math_library(context_project DEPS im2col math_function)
math_library(cos_sim_functor)
math_library(depthwise_conv)
math_library(sample_prob)
math_library(sampler DEPS generator)

# math_library(math_function DEPS blas dense_tensor tensor)
if(WITH_XPU)
  math_library(beam_search DEPS math_function beam_search_xpu)
else()
  math_library(beam_search DEPS math_function)
endif()

math_library(unpooling)
math_library(prelu)
math_library(bert_encoder_functor)
math_library(tree2col DEPS math_function)
