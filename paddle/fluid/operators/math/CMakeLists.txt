if(WITH_XPU)
  cc_library(
    beam_search_xpu
    SRCS beam_search_xpu.cc
    DEPS xpu_op_list)
endif()

# please add new math_library in alphabetical order
math_library(concat_and_split DEPS phi)
math_library(context_project DEPS phi)
math_library(cos_sim_functor)
math_library(depthwise_conv)
math_library(sample_prob)
math_library(sampler DEPS phi)

# math_library(phi DEPS phi tensor)

if(WITH_XPU)
  math_library(beam_search DEPS phi beam_search_xpu)
else()
  math_library(beam_search DEPS phi)
endif()

math_library(unpooling)
math_library(prelu)
math_library(bert_encoder_functor)
math_library(tree2col DEPS phi)

cc_test(
  selected_rows_functor_test
  SRCS selected_rows_functor_test.cc
  DEPS allocator phi)
cc_test(
  im2col_test
  SRCS im2col_test.cc
  DEPS phi)
cc_test(
  vol2col_test
  SRCS vol2col_test.cc
  DEPS phi)
cc_test(
  beam_search_test
  SRCS beam_search_test.cc
  DEPS beam_search)
if(WITH_GPU)
  nv_test(
    selected_rows_functor_gpu_test
    SRCS selected_rows_functor_test.cu.cc
    DEPS phi)
endif()
if(WITH_ROCM)
  hip_test(
    selected_rows_functor_gpu_test
    SRCS selected_rows_functor_test.cu.cc
    DEPS phi)
endif()
cc_test(
  concat_test
  SRCS concat_test.cc
  DEPS concat_and_split)

if(WITH_TESTING AND TEST im2col_test)
  set_tests_properties(im2col_test PROPERTIES TIMEOUT 120)
endif()
