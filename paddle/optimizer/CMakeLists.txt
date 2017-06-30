include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(OPITMIZER_SRCS
    adadelta_optimizer.cc
    adagrad_optimizer.cc
    adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optimizer.cc
  )

add_library(paddle_optimizer STATIC ${OPITMIZER_SRCS})
add_dependencies(paddle_optimizer paddle_proto ${external_project_dependencies})


if(WITH_TESTING)
  add_simple_unittest(serialization_test)
  add_simple_unittest(parameter_optimizer_test)
endif()
