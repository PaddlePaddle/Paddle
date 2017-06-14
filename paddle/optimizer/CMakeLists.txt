include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OPITMIZER_SRCS
    adadelta_optimizer.cc
    adagrad_optimizer.cc
    adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optimizer.cc
  )

add_library(optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(optimizer gen_proto_cpp)

add_simple_unittest(serialization_test)
add_simple_unittest(parameter_optimizer_test)
add_dependencies(parameter_optimizer_test optimizer)
