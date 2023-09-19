if(NOT (NOT WITH_PYTHON AND ON_INFER))
  cc_library(
    eager_tensor_operants
    SRCS eager_tensor_operants.cc
    DEPS final_dygraph_function)
endif()
