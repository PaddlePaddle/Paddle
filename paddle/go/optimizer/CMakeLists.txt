include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(optimizer_SRCS
    optimizer_factory.cc
    parameter_optimizer.cc
    training_ops.cc
  )

set(optimizer_Headers
    optimizer.h
    Tensor.h
    optimizer_factory.h
  #  optimizer_private.h
    parameter_optimizer.h
    regularizer.h
    regularizer_ops.h
    training_ops.h
  )

add_style_check_target(paddle_function ${optimizer_Headers})
add_style_check_target(paddle_function ${optimizer_SRCS})

add_library(optimizer SHARED optimizer.cc)
add_dependencies(optimizer gen_proto_cpp)
add_simple_unittest(optimizer_test)
add_simple_unittest(optimizer_factory_test)
# cc_test(optimizer_test
#   optimizer_SRCS
#   optimizer_test.cc
#   DEPS
#   optimizer
#   )
