cc_library(paddle_go_optimizer DEPS paddle_optimizer paddle_proto glog gflags protobuf)
target_link_libraries(paddle_go_optimizer stdc++ m)
go_library(paddle_pserver_cclient STATIC DEPS paddle_go_optimizer)
if(WITH_TESTING)
  # FIXME: this test requires pserver which is not managed by the test
  # we need some kind of e2e testing machanism.
  # add_subdirectory(test)
endif()
