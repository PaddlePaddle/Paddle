cc_library(paddle_go_optimizer DEPS paddle_optimizer paddle_proto glog gflags protobuf)
go_library(paddle_pserver_cclient STATIC DEPS paddle_go_optimizer)
if(WITH_TESTING)
    # TODO: add unit test
    #add_subdirectory(test)
endif()
