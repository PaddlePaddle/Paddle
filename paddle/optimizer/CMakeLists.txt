include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OPITMIZER_SRCS
    adadelta_optimizer.cc
    adagrad_optimizer.cc
    adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optmizer.cc
  )

set(OPITMIZER_HEADERS
    adadelta_optimizer.h
    adagrad_optimizer.h
    adam_optimizer.h
    lr_policy.h
    optimizer.h
    parameter_optimizer.h
    sgd_optimizer.h
    tensor.h
  )

add_library(optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(optimizer gen_proto_cpp)

add_simple_unittest(tensor_test)
add_simple_unittest(parameter_optimizer_test)
add_dependencies(parameter_optimizer_test optimizer)
