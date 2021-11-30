set(GRADIENT_ACCUMULATION_DEPS blas pten pten_api var_type_traits layer math_function)
#list(APPEND GRADIENT_ACCUMULATION_DEPS gradient_accumulator operator lod_tensor selected_rows selected_rows_functor)
if(WITH_XPU)
    list(APPEND GRADIENT_ACCUMULATION_DEPS xpu_op_list)
endif()

if(WITH_ASCEND_CL)
    list(APPEND GRADIENT_ACCUMULATION_DEPS npu_op_runner)
endif()

cc_library(gradient_accumulation SRCS gradient_accumulation.cc DEPS ${GRADIENT_ACCUMULATION_DEPS})
cc_library(accumulation_node SRCS accumulation_node.cc DEPS gradient_accumulation pten pten_api grad_node_info)
