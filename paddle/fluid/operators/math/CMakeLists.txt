if(WITH_ASCEND_CL)
  cc_library(
    beam_search_npu
    SRCS beam_search_npu.cc
    DEPS npu_op_runner)
endif()

if(WITH_XPU)
  cc_library(
    beam_search_xpu
    SRCS beam_search_xpu.cc
    DEPS xpu_op_list)
endif()

# please add new math_library in alphabetical order
if(WITH_ASCEND_CL)
  math_library(concat_and_split DEPS concat_and_split_functor npu_op_runner)
elseif(WITH_MLU)
  math_library(concat_and_split DEPS concat_and_split_functor mlu_baseop)
else()
  math_library(concat_and_split DEPS concat_and_split_functor)
endif()
math_library(context_project DEPS im2col math_function)
math_library(cos_sim_functor)
math_library(depthwise_conv)
math_library(sample_prob)
math_library(sampler DEPS generator)

# math_library(math_function DEPS blas dense_tensor tensor)
math_library(maxouting)

math_library(sequence_padding)
math_library(sequence_pooling DEPS math_function jit_kernel_helper)
math_library(sequence_scale)
if(WITH_ASCEND_CL)
  math_library(beam_search DEPS math_function beam_search_npu)
elseif(WITH_XPU)
  math_library(beam_search DEPS math_function beam_search_xpu)
else()
  math_library(beam_search DEPS math_function)
endif()
math_library(matrix_bit_code)

math_library(unpooling)
math_library(prelu)
math_library(bert_encoder_functor)
math_library(tree2col DEPS math_function)

cc_test(
  selected_rows_functor_test
  SRCS selected_rows_functor_test.cc
  DEPS allocator selected_rows_functor)
cc_test(
  im2col_test
  SRCS im2col_test.cc
  DEPS im2col)
cc_test(
  vol2col_test
  SRCS vol2col_test.cc
  DEPS vol2col)
cc_test(
  sequence_padding_test
  SRCS sequence_padding_test.cc
  DEPS sequence_padding)
cc_test(
  sequence_pooling_test
  SRCS sequence_pooling_test.cc
  DEPS sequence_pooling)
cc_test(
  beam_search_test
  SRCS beam_search_test.cc
  DEPS beam_search)
if(WITH_GPU)
  nv_test(
    selected_rows_functor_gpu_test
    SRCS selected_rows_functor_test.cu.cc
    DEPS selected_rows_functor math_function)
endif()
if(WITH_ROCM)
  hip_test(
    selected_rows_functor_gpu_test
    SRCS selected_rows_functor_test.cu.cc
    DEPS selected_rows_functor math_function)
endif()
cc_test(
  concat_test
  SRCS concat_test.cc
  DEPS concat_and_split)

if(WITH_TESTING AND TEST im2col_test)
  set_tests_properties(im2col_test PROPERTIES TIMEOUT 120)
endif()
