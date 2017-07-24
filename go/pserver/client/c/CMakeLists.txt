cc_library(paddle_go_optimizer DEPS paddle_optimizer paddle_proto glog gflags protobuf)
target_link_libraries(paddle_go_optimizer stdc++ m)

# Copy library to the required place.
# See: go/pserver/optimizer.go:
# // #cgo LDFLAGS: ${SRCDIR}/client/c/libpaddle_go_optimizer.a -lstdc++ -lm
add_custom_command(TARGET paddle_go_optimizer POST_BUILD
  COMMAND cp "${CMAKE_CURRENT_BINARY_DIR}/libpaddle_go_optimizer.a" "${CMAKE_CURRENT_SOURCE_DIR}"
  )

go_library(paddle_pserver_cclient STATIC DEPS paddle_go_optimizer)
if(WITH_TESTING)
  # FIXME: this test requires pserver which is not managed by the test
  # we need some kind of e2e testing machanism.
  # add_subdirectory(test)
endif()
