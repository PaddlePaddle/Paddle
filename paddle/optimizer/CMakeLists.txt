include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OPITMIZER_SRCS
    # adadelta_optimizer.cc
    # adagrad_optimizer.cc
    # adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optmizer.cc
    regularizer.cc
  )

set(OPITMIZER_Headers
    # adadelta_optimizer.h
    # adagrad_optimizer.h
    # adam_optimizer.h
    lr_policy.h
    optimizer.h
    parameter_optimizer.h
    regularizer.h
    sgd_optimizer.h
    Tensor.h
  )

add_library(optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(optimizer gen_proto_cpp)

add_simple_unittest(optimizer_test)
add_simple_unittest(optimizer_factory_test)
add_simple_unittest(Tensor_test)
