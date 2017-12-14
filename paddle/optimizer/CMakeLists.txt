set(OPITMIZER_SRCS
    adadelta_optimizer.cc
    adagrad_optimizer.cc
    adam_optimizer.cc
    optimizer.cc
    parameter_optimizer.cc
    sgd_optimizer.cc
  )

cc_library(paddle_optimizer STATIC SRCS ${OPITMIZER_SRCS} DEPS paddle_proto glog)
cc_test(serialization_test SRCS serialization_test.cc DEPS paddle_proto)
cc_test(parameter_optimizer_test SRCS parameter_optimizer_test.cc DEPS paddle_optimizer)
