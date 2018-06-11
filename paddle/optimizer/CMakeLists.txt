set(OPITMIZER_SRCS
    adadelta_optimizer.cc
    adagrad_optimizer.cc
    adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optimizer.cc
  )

add_library(paddle_optimizer ${OPITMIZER_SRCS})
target_link_libraries(paddle_optimizer paddle_proto glog)

if (WITH_TESTING)
    add_unittest(serialization_test serialization_test.cc)
    add_unittest(parameter_optimizer_test parameter_optimizer_test.cc)
endif()
