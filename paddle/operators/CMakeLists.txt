if(WITH_GPU)
    nv_library(add_op SRCS add_op.cc add_op.cu DEPS operator op_registry glog ddim)
else()
    cc_library(add_op SRCS add_op.cc DEPS operator op_registry glog ddim)
endif()
cc_test(add_op_test SRCS add_op_test.cc DEPS add_op)


cc_library(demo_op SRCS demo_op.cc DEPS operator op_registry)
cc_test(demo_op_test SRCS demo_op_test.cc DEPS demo_op)
