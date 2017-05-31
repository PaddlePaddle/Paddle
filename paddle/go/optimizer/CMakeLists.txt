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
  #  optimizer_private.h
    parameter_optimizer.h
    regularizer.h
  )

add_library(optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(optimizer gen_proto_cpp)

add_executable(optimizer_test optimizer_test.cpp)
add_dependencies(optimizer_test gen_proto_cpp gtest)
add_test(NAME optimizer_test COMMAND optimzier_test)
# add_simple_unittest(optimizer_test)
# add_simple_unittest(optimizer_factory_test)
