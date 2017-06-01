include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OPITMIZER_SRCS
    optimizer_factory.cc
    parameter_optimizer.cc
    regularizer.cc
  )

set(OPITMIZER_Headers
    optimizer.h
    Tensor.h
    optimizer_factory.h
    parameter_optimizer.h
    regularizer.h
  )

add_library(optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(optimizer gen_proto_cpp)

add_simple_unittest(optimizer_test)
add_simple_unittest(optimizer_factory_test)

